from __future__ import annotations

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from GeoStereo.controlnetvae import ControlNetVAEModel
from GeoStereo.dino_controlnetvae import DINOControlNetVAEModel
from GeoStereo.GeoStereo_pipeline import StableNormalPipeline




def cuda_sync_if_needed(device: torch.device | str):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def count_params_any(obj: Any):
    """
    Count parameters for diffusers pipeline / nn.Module / custom object.
    Return:
        total_params_m: float, in M
        detail: list[(name, params_m)]
    """
    seen_param_ids = set()
    detail = []

    def add_module(name: str, module: Any) -> int:
        if not hasattr(module, "parameters"):
            return 0

        n = 0
        try:
            for p in module.parameters(recurse=True):
                pid = id(p)
                if pid not in seen_param_ids:
                    seen_param_ids.add(pid)
                    n += p.numel()
        except Exception:
            return 0

        if n > 0:
            detail.append((name, n / 1e6))
        return n

    total = 0

    # diffusers-style components
    if hasattr(obj, "components") and isinstance(obj.components, dict):
        for name, module in obj.components.items():
            total += add_module(name, module)

    # nn.Module itself
    if isinstance(obj, torch.nn.Module):
        total += add_module("model", obj)

    # custom attributes
    try:
        obj_vars = vars(obj)
    except TypeError:
        obj_vars = {}

    for name, module in obj_vars.items():
        if isinstance(module, torch.nn.Module):
            total += add_module(name, module)

        if hasattr(module, "components") and isinstance(module.components, dict):
            for sub_name, sub_module in module.components.items():
                total += add_module(f"{name}.{sub_name}", sub_module)

    return total / 1e6, detail


def get_peak_mem_gb(device: torch.device | str) -> float:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0.0


def profile_flops_torch(run_fn, device: torch.device | str) -> float:
    """
    Profile FLOPs for one forward call.
    Unit: TFLOPs.

    Note:
    torch.profiler may under-count unsupported/custom CUDA ops,
    fused attention, xFormers kernels, or grid_sample.
    Use the same setting for all methods for fair comparison.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0

    cuda_sync_if_needed(device)

    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                run_fn()
                cuda_sync_if_needed(device)

        total_flops = 0
        for evt in prof.key_averages():
            flops = getattr(evt, "flops", None)
            if flops is not None:
                total_flops += flops

        return total_flops / 1e12

    except Exception as e:
        print(f"[WARN] FLOPs profiling failed: {repr(e)}")
        return 0.0


def write_csv(csv_path: str, records: list[dict]):
    if len(records) == 0:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def expand_lotus_unet_conv_in_4_to_8_tiny_extra(unet, extra_scale=1e-6):
    old_conv = unet.conv_in

    if old_conv.in_channels == 8:
        return unet

    assert old_conv.in_channels == 4, f"Expected 4ch Lotus conv_in, got {old_conv.in_channels}"

    new_conv = nn.Conv2d(
        in_channels=8,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :].copy_(old_conv.weight)
        new_conv.weight[:, 4:, :, :].copy_(old_conv.weight * extra_scale)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    unet.conv_in = new_conv
    unet.register_to_config(in_channels=8)
    return unet


def load_rgb_tensor(path):
    """
    path -> [1, 3, H, W], float32, range [0, 1]
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def save_normal_tensor(x, path):
    """
    x: [B, 3, H, W], range [-1, 1]
    save: RGB normal image, range [0, 255]
    """
    x = x.detach().float().cpu()
    x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    torchvision.utils.save_image(x, path)


def save_disp_tensor(disp, path, invalid_thr=1e-3, use_percentile=True):
    """
    disp: [B, 1, H, W]
    保存为 jet colormap 伪彩色视差图
    """
    disp = disp.detach().float().cpu()[0, 0].numpy()

    invalid_mask = disp <= invalid_thr

    if use_percentile:
        valid = disp[~invalid_mask]
        if valid.size > 0:
            dmin = np.percentile(valid, 1)
            dmax = np.percentile(valid, 99)
        else:
            dmin, dmax = 0.0, 1.0
    else:
        dmin = float(disp.min())
        dmax = float(disp.max())

    disp_norm = (disp - dmin) / (dmax - dmin + 1e-6)
    disp_norm = np.clip(disp_norm, 0.0, 1.0)

    disp_u8 = (disp_norm * 255.0).astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
    disp_color[invalid_mask] = 0
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    Image.fromarray(disp_color).save(path)


def resolve_path(path_str, root=None):
    """
    如果 path_str 是绝对路径，直接返回；
    如果是相对路径，则拼接 root。
    """
    path_str = os.path.expanduser(os.path.expandvars(path_str))
    p = Path(path_str)

    if p.is_absolute():
        return str(p)

    if root is None:
        return str(p)

    root = Path(os.path.expanduser(os.path.expandvars(root)))
    return str(root / p)


def read_pair_txt(txt_path, root=None):
    """
    txt 每一行格式:
        relative_left_path relative_right_path
    """
    pairs = []
    txt_path = Path(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if len(line) == 0:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line {line_idx} in {txt_path}: "
                    f"expected two paths separated by space, got: {line}"
                )

            left_rel = parts[0]
            right_rel = parts[1]

            left_path = resolve_path(left_rel, root=root)
            right_path = resolve_path(right_rel, root=root)

            pairs.append((left_path, right_path))

    if len(pairs) == 0:
        raise ValueError(f"No valid image pairs found in {txt_path}")

    return pairs


def make_safe_name(path):
    stem = Path(path).stem
    stem = stem.replace(" ", "_")
    stem = stem.replace("/", "_")
    stem = stem.replace("\\", "_")
    return stem


# ============================================================
# Load pipeline
# ============================================================

def build_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.fp16:
        mp_dtype = torch.float16
    else:
        mp_dtype = torch.float32

    pipe_args = SimpleNamespace(
        cfg_yaml=args.foundation_cfg,
        ckpt_dir=args.foundation_ckpt,
        mono=False,
        train_iters=args.train_iters,
        max_disp=args.max_disp,
        t_start=args.t_start,
    )

    print("[INFO] loading tokenizer/text_encoder/vae/scheduler...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    print(f"[INFO] load finetuned checkpoint from: {args.finetuned_ckpt}")
    unet = UNet2DConditionModel.from_pretrained(args.finetuned_ckpt, subfolder="unet")
    rgb_ctl = ControlNetVAEModel.from_pretrained(args.finetuned_ckpt, subfolder="controlnet")
    dino_ctl = DINOControlNetVAEModel.from_pretrained(args.finetuned_ckpt, subfolder="dino_controlnet")

    print("[INFO] building StableNormalPipeline...")
    pipe = StableNormalPipeline(
        args=pipe_args,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=rgb_ctl,
        dino_controlnet=dino_ctl,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
        default_denoising_steps=args.denoising_steps,
        default_processing_resolution=args.processing_resolution,
        prompt="",
        empty_text_embedding=None,
    )

    print(f"[INFO] load FoundationStereo from: {args.foundation_ckpt}")
    ckpt = torch.load(args.foundation_ckpt, map_location="cpu")
    missing_keys, unexpected_keys = pipe.FoundationStereo.load_state_dict(
        ckpt["model"],
        strict=False,
    )
    print("[INFO] FoundationStereo missing_keys:", missing_keys)
    print("[INFO] FoundationStereo unexpected_keys:", unexpected_keys)

    pipe = pipe.to(device)

    # FoundationStereo 建议 fp32
    pipe.FoundationStereo = pipe.FoundationStereo.to(device=device, dtype=torch.float32)

    try:
        pipe.FoundationStereo.feature = pipe.FoundationStereo.feature.to(
            device=device,
            dtype=torch.float32,
        )
        pipe.FoundationStereo.feature.dino = pipe.FoundationStereo.feature.dino.to(
            device=device,
            dtype=torch.float32,
        )
    except Exception as e:
        print("[Warning] failed to move FoundationStereo feature/dino explicitly:", e)

    pipe.unet.to(dtype=mp_dtype)
    pipe.controlnet.to(dtype=mp_dtype)
    pipe.dino_controlnet.to(dtype=mp_dtype)
    pipe.vae.to(dtype=mp_dtype)
    pipe.text_encoder.to(dtype=mp_dtype)

    pipe.unet.eval()
    pipe.controlnet.eval()
    pipe.dino_controlnet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.FoundationStereo.eval()

    return pipe, device


# ============================================================
# Inference helpers
# ============================================================

@torch.inference_mode()
def run_geostereo_once(pipe, device, left_img, right_img, control_scale=0.2):
    left = load_rgb_tensor(left_img).to(device)
    right = load_rgb_tensor(right_img).to(device)

    out = pipe(
        image1=left,
        image2=right,
        controlnet_conditioning_scale=control_scale,
        output_type="pt",
    )
    return out


@torch.inference_mode()
def run_geostereo_timed(
    pipe,
    device,
    left_img,
    right_img,
    control_scale=0.2,
    profile_flops=False,
):
    """
    Return:
        out
        infer_time: seconds
        peak_mem_gb
        flops_t
    """
    torch_device = torch.device(device)

    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch_device)

    cuda_sync_if_needed(torch_device)
    t0 = time.perf_counter()

    out = run_geostereo_once(
        pipe=pipe,
        device=device,
        left_img=left_img,
        right_img=right_img,
        control_scale=control_scale,
    )

    cuda_sync_if_needed(torch_device)
    infer_time = time.perf_counter() - t0

    peak_mem_gb = get_peak_mem_gb(torch_device)

    flops_t = 0.0
    if profile_flops:
        def run_once_for_flops():
            return run_geostereo_once(
                pipe=pipe,
                device=device,
                left_img=left_img,
                right_img=right_img,
                control_scale=control_scale,
            )

        # extra call only for FLOPs, not included in infer_time
        flops_t = profile_flops_torch(run_once_for_flops, torch_device)

    return out, infer_time, peak_mem_gb, flops_t


def save_outputs(out, save_dir, left_img, right_img):
    os.makedirs(save_dir, exist_ok=True)

    pred = out.prediction

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).permute(0, 3, 1, 2)

    pred = pred.detach().cpu()
    save_normal_tensor(pred, os.path.join(save_dir, "pred_normal.png"))

    if getattr(out, "normal_init", None) is not None:
        save_normal_tensor(
            out.normal_init.detach().cpu(),
            os.path.join(save_dir, "normal_init.png"),
        )

    if getattr(out, "disp_pred", None) is not None:
        save_disp_tensor(
            out.disp_pred,
            os.path.join(save_dir, "disp_pred.png"),
        )

    with open(os.path.join(save_dir, "pair.txt"), "w", encoding="utf-8") as f:
        f.write(f"{left_img} {right_img}\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument(
        "--pair_txt",
        type=str,
        default="/file_system/goosefsx/whu/users/hong.li/qizhe/test.txt",
        help="txt file. Each line: left_image_path right_image_path",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/file_system/goosefsx/public_data/DepthAnything3/qizhe/datasets/IRS/IRS",
        help="Root directory for relative paths in pair_txt.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./infer_outputs_txt",
        help="root output directory",
    )

    # model paths
    parser.add_argument(
        "--base_model",
        type=str,
        default="/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/lotus",
    )

    parser.add_argument(
        "--foundation_cfg",
        type=str,
        default="/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/cfg.yaml",
    )

    parser.add_argument(
        "--foundation_ckpt",
        type=str,
        default="/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/model_best_bp2-001.pth",
    )

    parser.add_argument(
        "--finetuned_ckpt",
        type=str,
        default="/file_system/goosefsx/whu/users/hong.li/qizhe/diffusion-e2e-ft-main/training/root1/ckpt/checkpoint-1400",
    )

    # pipeline args
    parser.add_argument("--train_iters", type=int, default=22)
    parser.add_argument("--max_disp", type=int, default=416)
    parser.add_argument("--t_start", type=int, default=999)
    parser.add_argument("--denoising_steps", type=int, default=1)
    parser.add_argument("--processing_resolution", type=int, default=768)
    parser.add_argument("--control_scale", type=float, default=0.2)

    # precision
    parser.add_argument("--fp16", action="store_true")

    # efficiency
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--method_name", type=str, default="GeoStereo")
    parser.add_argument("--profile_flops", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    pairs = read_pair_txt(args.pair_txt, root=args.root)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"[INFO] loaded {len(pairs)} image pairs from {args.pair_txt}")

    pipe, device = build_pipeline(args)
    torch_device = torch.device(device)

    params_m, params_detail = count_params_any(pipe)
    print("=" * 80)
    print(f"[PARAMS] Total Params: {params_m:.2f}M")
    for name, n in params_detail:
        print(f"[PARAMS] {name}: {n:.2f}M")
    print("=" * 80)

    timing_records = []
    total_infer_time = 0.0
    valid_count = 0
    max_peak_mem_gb_all = 0.0
    profiled_flops_t = 0.0

    timing_csv_path = os.path.join(args.save_dir, "timing_summary.csv")

    for idx, (left_img, right_img) in enumerate(pairs):
        left_name = make_safe_name(left_img)
        pair_save_dir = os.path.join(args.save_dir, f"{idx:06d}_{left_name}")

        print("=" * 80)
        print(f"[INFO] pair {idx + 1}/{len(pairs)}")
        print(f"[INFO] left : {left_img}")
        print(f"[INFO] right: {right_img}")
        print(f"[INFO] save : {pair_save_dir}")

        try:
            do_profile_flops = args.profile_flops and (profiled_flops_t == 0.0)

            out, infer_time, peak_mem_gb, flops_t = run_geostereo_timed(
                pipe=pipe,
                device=device,
                left_img=left_img,
                right_img=right_img,
                control_scale=args.control_scale,
                profile_flops=do_profile_flops,
            )

            if flops_t > 0:
                profiled_flops_t = flops_t

            shown_flops = flops_t if flops_t > 0 else profiled_flops_t

            max_peak_mem_gb_all = max(max_peak_mem_gb_all, peak_mem_gb)

            save_outputs(
                out=out,
                save_dir=pair_save_dir,
                left_img=left_img,
                right_img=right_img,
            )

            total_infer_time += infer_time
            valid_count += 1

            avg_infer_time = total_infer_time / valid_count
            avg_fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0.0
            fps = 1.0 / infer_time if infer_time > 0 else 0.0

            print(
                f"[EFF] idx={idx:06d} | "
                f"Time={infer_time:.4f}s | "
                f"Params={params_m:.2f}M | "
                f"Mem={peak_mem_gb:.2f}GB | "
                f"FLOPs={shown_flops:.3f}T | "
                f"Steps={args.denoising_steps} | "
                f"FPS={fps:.2f} | "
                f"avg_Time={avg_infer_time:.4f}s | "
                f"avg_FPS={avg_fps:.2f}"
            )

            timing_records.append(
                {
                    "idx": idx,
                    "left_path": left_img,
                    "right_path": right_img,
                    "save_dir": pair_save_dir,
                    "infer_time_sec": infer_time,
                    "fps": fps,
                    "params_m": params_m,
                    "peak_mem_gb": peak_mem_gb,
                    "flops_t": shown_flops,
                    "denoising_steps": args.denoising_steps,
                    "processing_resolution": args.processing_resolution,
                    "control_scale": args.control_scale,
                    "fp16": args.fp16,
                    "avg_infer_time_sec_until_now": avg_infer_time,
                    "avg_fps_until_now": avg_fps,
                    "status": "ok",
                    "error": "",
                }
            )

            write_csv(timing_csv_path, timing_records)
            print(f"[OK] saved results to: {pair_save_dir}")

        except Exception as e:
            print(f"[ERROR] failed on pair {idx}: {left_img} {right_img}")
            print(e)

            with open(os.path.join(args.save_dir, "failed.txt"), "a", encoding="utf-8") as f:
                f.write(f"{idx}\t{left_img}\t{right_img}\t{repr(e)}\n")

            timing_records.append(
                {
                    "idx": idx,
                    "left_path": left_img,
                    "right_path": right_img,
                    "save_dir": pair_save_dir,
                    "infer_time_sec": "",
                    "fps": "",
                    "params_m": params_m,
                    "peak_mem_gb": "",
                    "flops_t": "",
                    "denoising_steps": args.denoising_steps,
                    "processing_resolution": args.processing_resolution,
                    "control_scale": args.control_scale,
                    "fp16": args.fp16,
                    "avg_infer_time_sec_until_now": "",
                    "avg_fps_until_now": "",
                    "status": "failed",
                    "error": repr(e),
                }
            )
            write_csv(timing_csv_path, timing_records)

    print("[INFO] all done. results saved in:", args.save_dir)
    print(f"[INFO] timing csv -> {timing_csv_path}")

    if valid_count > 0:
        avg_infer_time = total_infer_time / valid_count
        avg_fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0.0

        timing_average_path = os.path.join(args.save_dir, "timing_average.txt")
        efficiency_summary_path = os.path.join(args.save_dir, "efficiency_summary.csv")

        with open(timing_average_path, "w", encoding="utf-8") as f:
            f.write(f"valid_count: {valid_count}\n")
            f.write(f"avg_infer_time_sec: {avg_infer_time:.6f}\n")
            f.write(f"avg_fps: {avg_fps:.6f}\n")
            f.write(f"params_m: {params_m:.6f}\n")
            f.write(f"max_peak_mem_gb: {max_peak_mem_gb_all:.6f}\n")
            f.write(f"profiled_flops_t: {profiled_flops_t:.6f}\n")
            f.write(f"denoising_steps: {args.denoising_steps}\n")
            f.write(f"processing_resolution: {args.processing_resolution}\n")
            f.write(f"control_scale: {args.control_scale}\n")
            f.write(f"fp16: {args.fp16}\n")

        with open(efficiency_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Method",
                    "Time(s)",
                    "Params(M)",
                    "Mem.(GB)",
                    "FLOPs(T)",
                    "Steps",
                    "ProcessingRes",
                    "ControlScale",
                    "FP16",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Method": args.method_name,
                    "Time(s)": f"{avg_infer_time:.4f}",
                    "Params(M)": f"{params_m:.2f}",
                    "Mem.(GB)": f"{max_peak_mem_gb_all:.2f}",
                    "FLOPs(T)": f"{profiled_flops_t:.3f}",
                    "Steps": args.denoising_steps,
                    "ProcessingRes": args.processing_resolution,
                    "ControlScale": args.control_scale,
                    "FP16": args.fp16,
                }
            )

        print("=" * 80)
        print(f"[SUMMARY] valid_count={valid_count}")
        print(f"[SUMMARY] avg_Time(s)={avg_infer_time:.6f}")
        print(f"[SUMMARY] avg_FPS={avg_fps:.4f}")
        print(f"[SUMMARY] Params(M)={params_m:.2f}")
        print(f"[SUMMARY] Mem.(GB)={max_peak_mem_gb_all:.2f}")
        print(f"[SUMMARY] FLOPs(T)={profiled_flops_t:.3f}")
        print(f"[SUMMARY] Steps={args.denoising_steps}")
        print(f"[SUMMARY] efficiency csv -> {efficiency_summary_path}")
    else:
        print("[SUMMARY] no valid pairs were processed.")


if __name__ == "__main__":
    main()
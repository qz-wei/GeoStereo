import os
import sys
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from GeoStereo.controlnetvae import ControlNetVAEModel
from GeoStereo.dino_controlnetvae import DINOControlNetVAEModel
from GeoStereo.GeoStereo_pipeline import StableNormalPipeline

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
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x


def save_normal_tensor(x, path):
    """
    x: [B,3,H,W], range [-1,1]
    """
    x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    torchvision.utils.save_image(x, path)

def save_disp_tensor(disp, path, invalid_thr=1e-3, use_percentile=True):
    """
    disp: [B,1,H,W]
    保存为 jet colormap 伪彩色视差图
    """
    disp = disp.detach().float().cpu()[0, 0].numpy()  # [H,W]

    invalid_mask = disp <= invalid_thr

    # 归一化范围：建议用 percentile，更稳一点
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

    # OpenCV 默认输出是 BGR
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

    # 无效区域置黑（可选）
    disp_color[invalid_mask] = 0

    # 转成 RGB 再保存
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)
    Image.fromarray(disp_color).save(path)


def main():
    # ---------------------------
    # 这里改成你的真实路径
    # ---------------------------
    BASE_MODEL = "/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/lotus"
    FOUNDATION_CFG = "/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/cfg.yaml"
    FOUNDATION_CKPT = "/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/model_best_bp2-001.pth"
    #LEFT_IMG = "/file_system/goosefsx/whu/users/hong.li/qizhe/vis_data/left/Explorer_HD2K_SN36965281_17-33-52_left.png"
    #RIGHT_IMG = "/file_system/goosefsx/whu/users/hong.li/qizhe/vis_data/right/Explorer_HD2K_SN36965281_17-33-52_right.png"
    

    LEFT_IMG = "/file_system/goosefsx/whu/users/hong.li/qizhe/diffusion-e2e-ft-main/training/dataloaders/check_mixed_loader_out/irs/samples/irs_sample_3_left_rgb.png"
    RIGHT_IMG = "/file_system/goosefsx/whu/users/hong.li/qizhe/diffusion-e2e-ft-main/training/dataloaders/check_mixed_loader_out/irs/samples/irs_sample_3_right_rgb.png"
    

    #LEFT_IMG= "/file_system/goosefsx/whu/users/hong.li/qizhe/vis_data/left/Explorer_HD2K_SN36965281_21-36-14_left.png"
    #RIGHT_IMG = "/file_system/goosefsx/whu/users/hong.li/qizhe/vis_data/right/Explorer_HD2K_SN36965281_21-36-14_right.png"
    SAVE_DIR = "./infer_outputs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp_dtype = torch.float32 if torch.cuda.is_available() else torch.float32

    # ---------------------------
    # 构造 pipeline 所需 args
    # 这些字段要和你 pipeline 里会访问到的字段对上
    # ---------------------------
    args = SimpleNamespace(
        cfg_yaml=FOUNDATION_CFG,
        ckpt_dir=FOUNDATION_CKPT,
        mono=False,
        train_iters=15,
        max_disp=416,
        t_start=999,  
    )

    # ---------------------------
    # 1. 加载公共模块
    # ---------------------------
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    # ---------------------------
    # 2. 加载 UNet / ControlNets
    #    优先加载你 finetune 后的 checkpoint
    # ---------------------------
    

    FINETUNED_CKPT = "/file_system/goosefsx/whu/users/hong.li/qizhe/diffusion-e2e-ft-main/training/root1/ckpt/checkpoint-1400"
    print(f"[INFO] load finetuned checkpoint from: {FINETUNED_CKPT}")
    unet = UNet2DConditionModel.from_pretrained(FINETUNED_CKPT, subfolder="unet")
    rgb_ctl = ControlNetVAEModel.from_pretrained(FINETUNED_CKPT, subfolder="controlnet")
    dino_ctl = DINOControlNetVAEModel.from_pretrained(FINETUNED_CKPT, subfolder="dino_controlnet")

    # ---------------------------
    # 3. 建 pipeline
    # ---------------------------
    pipe = StableNormalPipeline(
        args=args,
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
        default_denoising_steps=1,
        default_processing_resolution=768,
        prompt="",
        empty_text_embedding=None,
    )

    print(f"[INFO] load FoundationStereo from: {FOUNDATION_CKPT}")
    ckpt = torch.load(FOUNDATION_CKPT, map_location="cpu")
    missing_keys, unexpected_keys = pipe.FoundationStereo.load_state_dict(
        ckpt["model"], strict=False
    )
    print("[INFO] FoundationStereo missing_keys:", missing_keys)
    print("[INFO] FoundationStereo unexpected_keys:", unexpected_keys)


    pipe = pipe.to(device)

    # FoundationStereo 通常建议保持 fp32
    pipe.FoundationStereo = pipe.FoundationStereo.to(device=device, dtype=torch.float32)
    try:
        pipe.FoundationStereo.feature = pipe.FoundationStereo.feature.to(device=device, dtype=torch.float32)
        pipe.FoundationStereo.feature.dino = pipe.FoundationStereo.feature.dino.to(device=device, dtype=torch.float32)
    except Exception:
        pass

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

    # ---------------------------
    # 6. 读图
    # ---------------------------
    left = load_rgb_tensor(LEFT_IMG).to(device)
    right = load_rgb_tensor(RIGHT_IMG).to(device)

    # ---------------------------
    # 7. 推理
    # ---------------------------
    with torch.no_grad():
        out = pipe(
            image1=left,
            image2=right,
            controlnet_conditioning_scale=0.2,
            output_type="pt",   # 保持 torch tensor 输出
        )

    pred = out.prediction
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).permute(0, 3, 1, 2)

    pred = pred.detach().cpu()

    save_normal_tensor(pred, os.path.join(SAVE_DIR, "pred_normal.png"))

    if out.normal_init is not None:
        save_normal_tensor(out.normal_init.detach().cpu(), os.path.join(SAVE_DIR, "normal_init.png"))

    if out.disp_pred is not None:
        save_disp_tensor(out.disp_pred, os.path.join(SAVE_DIR, "disp_pred.png"))

    print("[INFO] done. results saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect 
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from .depthanything_head import DepthAnythingFeature

from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.schedulers import (
    DDIMScheduler
)

from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
from omegaconf import OmegaConf
import cv2
import contextlib
from GeoStereo.core.foundation_stereo import FoundationStereo
from einops import rearrange  
import torchvision
import time

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DINO_PATCH = 14  # dinov2 patch size (assert H,W multiple of 14)

def normalize_normals(n: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return n / (torch.norm(n, dim=1, keepdim=True) + eps)

def warp_right_to_left(
    right_bchw: torch.Tensor,
    disp_b1hw: torch.Tensor,
    sign: str = "minus",   # "minus": x_r = x_l - d  (common). "plus": x_r = x_l + d
    align_corners: bool = True,
) -> torch.Tensor:
    B, C, H, W = right_bchw.shape
    xs = torch.linspace(0, W - 1, W, device=right_bchw.device, dtype=right_bchw.dtype)
    ys = torch.linspace(0, H - 1, H, device=right_bchw.device, dtype=right_bchw.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # H,W
    X = X[None, None].expand(B, 1, H, W)
    Y = Y[None, None].expand(B, 1, H, W)

    if sign == "minus":
        Xr = X - disp_b1hw
    elif sign == "plus":
        Xr = X + disp_b1hw
    else:
        raise ValueError("sign must be 'minus' or 'plus'")

    grid_x = 2.0 * (Xr / (W - 1.0)) - 1.0
    grid_y = 2.0 * (Y  / (H - 1.0)) - 1.0
    grid = torch.cat([grid_x, grid_y], dim=1)      # [B,2,H,W]
    grid = grid.permute(0, 2, 3, 1).contiguous()   # [B,H,W,2]

    warped = F.grid_sample(
        right_bchw, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return warped

@torch.no_grad()
def disp_to_normal_points_no_intrinsics(
    disp: torch.Tensor,
    max_disp: float,
    eps: float = 1e-6,
    use_log: bool = True,
    use_inverse: bool = False,
    smooth_ks: int = 5,
    focal_scale: float = 1.5,
    flip_x: bool = True,
    flip_y: bool = False,
    flip_z: bool = False,
    nz_positive: bool = True,
) -> torch.Tensor:
    if disp.dim() == 3:
        disp = disp.unsqueeze(1)
    disp = disp.float()
    B, _, H, W = disp.shape

    disp = torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=eps)

    if use_log:
        d01 = (disp / float(max_disp)).clamp(min=eps, max=1.0)
        Z = -torch.log(d01)
    else:
        Z = disp
        if use_inverse:
            Z = 1.0 / Z

    if smooth_ks and smooth_ks > 1:
        Z = F.avg_pool2d(Z, kernel_size=smooth_ks, stride=1, padding=smooth_ks // 2)

    xs = torch.linspace(-1, 1, W, device=disp.device, dtype=disp.dtype)
    ys = torch.linspace(-1, 1, H, device=disp.device, dtype=disp.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    X = X / (focal_scale + eps)
    Y = Y / (focal_scale + eps)

    gx = X.view(1, 1, H, W).expand(B, 1, H, W)
    gy = Y.view(1, 1, H, W).expand(B, 1, H, W)
    gz = torch.ones_like(gx)

    Px = gx * Z
    Py = gy * Z
    Pz = gz * Z

    def cdiff_x(t):
        return F.pad(t[:, :, :, 2:] - t[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")

    def cdiff_y(t):
        return F.pad(t[:, :, 2:, :] - t[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")

    dPx_dx, dPy_dx, dPz_dx = cdiff_x(Px), cdiff_x(Py), cdiff_x(Pz)
    dPx_dy, dPy_dy, dPz_dy = cdiff_y(Px), cdiff_y(Py), cdiff_y(Pz)

    Nx = dPy_dx * dPz_dy - dPz_dx * dPy_dy
    Ny = dPz_dx * dPx_dy - dPx_dx * dPz_dy
    Nz = dPx_dx * dPy_dy - dPy_dx * dPx_dy
    n = torch.cat([Nx, Ny, Nz], dim=1)

    if flip_x: n[:, 0:1] *= -1.0
    if flip_y: n[:, 1:2] *= -1.0
    if flip_z: n[:, 2:3] *= -1.0

    n = n / (torch.norm(n, dim=1, keepdim=True) + eps)

    if nz_positive:
        flip = (n[:, 2:3] < 0).float()
        n = n * (1.0 - 2.0 * flip)

    n = torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    return n

def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)
  W_resize = round_by_divider(W)
  if H_resize>max_H or W_resize>max_W:
    if H_resize>W_resize:
      W_resize = round_by_divider(W_resize*max_H/H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize*max_W/W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        if x.shape[0] in (1, 3):
            return x.unsqueeze(0)
        if x.shape[-1] in (1, 3):
            return x.permute(2, 0, 1).unsqueeze(0).contiguous()
        raise ValueError(f"Unsupported 3D shape: {tuple(x.shape)}")
    if x.dim() == 4:
        if x.shape[1] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported 4D shape: {tuple(x.shape)}")
    raise ValueError(f"Unsupported dim={x.dim()} for image-like tensor")


def img_to_01(img: torch.Tensor) -> torch.Tensor:
    """Robustly map image to [0,1] regardless of uint8 / float(0..1) / float(0..255)."""
    img = ensure_bchw(img)
    if img.dtype == torch.uint8:
        return (img.float() / 255.0).clamp(0, 1)
    mx = float(img.detach().max().cpu())
    mn = float(img.detach().min().cpu())
    if mx > 1.5:
        return (img.float() / 255.0).clamp(0, 1)
    if mn < -0.1:
        return ((img.float() + 1.0) * 0.5).clamp(0, 1)
    return img.float().clamp(0, 1)


def img_to_m11(img: torch.Tensor) -> torch.Tensor:
    """Map input image to [-1,1]. Supports uint8, [0,1], [0,255], and already-[-1,1] tensors."""
    img = ensure_bchw(img)
    if img.dtype == torch.uint8:
        return (img.float() / 127.5 - 1.0).clamp(-1, 1)
    img = img.float()
    mx = float(img.detach().max().cpu())
    mn = float(img.detach().min().cpu())
    if mn >= -1.05 and mx <= 1.05 and mn < -0.1:
        return img.clamp(-1, 1)
    if mx > 1.5:
        return (img / 127.5 - 1.0).clamp(-1, 1)
    return (img * 2.0 - 1.0).clamp(-1, 1)


def m11_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def normalize_image_imagenet(img: torch.Tensor) -> torch.Tensor:
    img01 = img_to_01(img)
    tf = torchvision.transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD), inplace=False)
    return tf(img01).contiguous()

def _cuda_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _log_dt(name, t_start, prefix="[TIME]"):
    t_end = _cuda_time()
    dt = t_end - t_start
    print(f"{prefix} {name:<32s}: {dt*1000:.2f} ms")
    return t_end


@dataclass
class StableNormalOutput(BaseOutput):
    prediction: Union[np.ndarray, torch.Tensor]
    latent: Optional[torch.Tensor] = None
    gaus_noise: Optional[torch.Tensor] = None
    disp_pred: Optional[torch.Tensor] = None
    normal_init: Optional[torch.Tensor] = None

class StableNormalPipeline(StableDiffusionControlNetPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet,
        dino_controlnet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker=True,
        default_denoising_steps=1,
        default_processing_resolution=768,
        prompt="a normal map",
        empty_text_embedding=None,
        args=None,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

        self.register_modules(
            dino_controlnet=dino_controlnet,
        )
        self.args = args
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.dino_image_processor = lambda x: x / 127.5 -1.
        cfg = OmegaConf.load(args.cfg_yaml)
        for k, v in vars(args).items():
            cfg[k] = v
        margs = OmegaConf.create(cfg)
        self.margs = margs
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.prompt = prompt
        self.prompt_embeds = None
        self.empty_text_embedding = empty_text_embedding
        depth_any = DepthAnythingFeature(
            encoder="vitl",
            tokens_out_dim=None,
            resize_to=None,
        )
        self.mono = args.mono
        self.FoundationStereo = FoundationStereo(self.margs, depth_any)
        self.train_iters = int(getattr(args, "train_iters", 10))
        #self.register_modules(args=args)

    def force_fp32(self, device=None):
        """Force every inference module used by this pipeline to float32.

        This is useful when eval.py or Diffusers has previously converted modules to fp16/bf16.
        The current training script was run with mixed_precision='no', so this keeps inference
        numerically aligned with training.
        """
        if device is None:
            device = self._execution_device

        for name in ("vae", "text_encoder", "unet", "controlnet", "dino_controlnet"):
            module = getattr(self, name, None)
            if module is not None:
                module.to(device=device, dtype=torch.float32)

        if hasattr(self, "FoundationStereo") and self.FoundationStereo is not None:
            self.FoundationStereo = self.FoundationStereo.to(device=device, dtype=torch.float32)
            try:
                self.FoundationStereo.feature = self.FoundationStereo.feature.to(device=device, dtype=torch.float32)
                self.FoundationStereo.feature.dino = self.FoundationStereo.feature.dino.to(device=device, dtype=torch.float32)
            except Exception:
                pass

        return self

    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        image1,
        image2,
        prompt=None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_images_per_prompt: Optional[int] = 1,
        generator=None,
        controlnet_conditioning_scale: Union[float, List[float]] = None,
        output_type: str = "np",
        output_uncertainty: bool = False,
        output_latent: bool = False,
        return_dict: bool = True,
    ):
        """
        Inference path aligned with the current training script.

        Training does:
            unet_input = cat([VAE(left_rgb), VAE(normal_init)], dim=1)
            timestep = 999
            rgb_controlnet input = VAE(warp_right_to_left(right_rgb, disp))
            dino_controlnet input = FoundationStereo left_tokens
            unet predicts normal latent directly in one forward pass.

        Therefore this pipeline intentionally does NOT run a DDIM denoising loop.
        """
        device = self._execution_device
        self.force_fp32(device)
        t_all = _cuda_time()

        # From this point on, all model inputs/conditions are kept in fp32.
        fp32_dtype = torch.float32

        if not torch.is_tensor(image1) or not torch.is_tensor(image2):
            raise ValueError("为了和训练代码对齐，这里建议直接喂 torch Tensor 的 left/right。")
        t_fixed = int(getattr(self.args, "t_fixed", 999))
        max_disp = float(getattr(self.args, "max_disp", 416))
        train_iters = int(getattr(self, "train_iters", getattr(self.args, "train_iters", 15)))
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = float(getattr(self.args, "control_scale", 0.2))
        warp_sign = getattr(self.args, "warp_sign", "minus")

        # -----------------------------
        # A) input -> [-1,1], adaptive resize
        # -----------------------------
        t0 = _cuda_time()
        left_m11 = img_to_m11(image1).to(device, non_blocking=True)
        right_m11 = img_to_m11(image2).to(device, non_blocking=True)

        B_orig, _, H_orig, W_orig = left_m11.shape
        original_resolution = (H_orig, W_orig)

        divider = int(getattr(self.args, "infer_divider", 32))
        max_res = int(getattr(self.args, "infer_max_res", 2240))
        H_resize, W_resize = get_resize_keep_aspect_ratio(
            H_orig, W_orig, divider=divider, max_H=max_res, max_W=max_res
        )

        if (H_resize, W_resize) != (H_orig, W_orig):
            left_m11 = F.interpolate(left_m11, size=(H_resize, W_resize), mode="bicubic", align_corners=False)
            right_m11 = F.interpolate(right_m11, size=(H_resize, W_resize), mode="bicubic", align_corners=False)

        B, _, H, W = left_m11.shape
        _log_dt("input -> m11 & adaptive resize", t0)

        # -----------------------------
        # B) stereo normalization, exactly like training
        # -----------------------------
        t1 = _cuda_time()
        left_01 = m11_to_01(left_m11)
        right_01 = m11_to_01(right_m11)
        left_stereo = normalize_image_imagenet(left_01)
        right_stereo = normalize_image_imagenet(right_01)
        print(f"Adaptive Resized Shapes: left {left_stereo.shape}, right {right_stereo.shape}")
        _log_dt("normalize_image_imagenet", t1)

        try:
            if hasattr(self.FoundationStereo, "args") and hasattr(self.FoundationStereo.args, "mixed_precision"):
                self.FoundationStereo.args.mixed_precision = False
        except Exception:
            pass

        no_autocast = torch.autocast("cuda", enabled=False) if device.type == "cuda" else contextlib.nullcontext()

        # -----------------------------
        # C) FoundationStereo: use returned left_tokens when available
        # -----------------------------
        t2 = _cuda_time()
        with no_autocast:
            _, disp_preds, left_tokens = self.FoundationStereo(
                left_stereo.float(),
                right_stereo.float(),
                iters=train_iters,
                test_mode=False,
            )
            disp_pred = disp_preds[-1].float()
            if disp_pred.dim() == 3:
                disp_pred = disp_pred.unsqueeze(1)
            disp_pred_safe = disp_pred.detach().clamp(min=1e-3, max=max_disp)
        _log_dt("FoundationStereo forward", t2)

        # -----------------------------
        # D) disparity -> clean normal_init, exactly matching training signs
        # -----------------------------
        t3 = _cuda_time()
        normal_init = disp_to_normal_points_no_intrinsics(
            disp_pred_safe.detach(),
            max_disp=max_disp,
            use_log=True,
            smooth_ks=5,
            focal_scale=1.5,
            flip_x=False,
            flip_y=False,
            flip_z=False,
        )
        valid_normal = (disp_pred_safe > 1e-3).float()
        normal_init = normalize_normals(normal_init) * valid_normal
        normal_init = normal_init.clamp(-1.0, 1.0)
        _log_dt("disp_to_normal_points_no_intrinsics", t3)

        # -----------------------------
        # E) VAE latents: cat([left_rgb_latent, init_normal_latent]) for 8ch UNet
        # -----------------------------
        t4 = _cuda_time()
        vae_dtype = fp32_dtype
        unet_dtype = fp32_dtype
        with no_autocast:
            # Use mode for deterministic inference. If you want stochastic VAE sampling, change mode() to sample().
            rgb_latent = self.vae.encode(left_m11.to(dtype=vae_dtype)).latent_dist.mode() * self.vae.config.scaling_factor
            init_latent = self.vae.encode(normal_init.to(dtype=vae_dtype)).latent_dist.mode() * self.vae.config.scaling_factor
        rgb_latent = rgb_latent.to(device=device, dtype=fp32_dtype)
        init_latent = init_latent.to(device=device, dtype=fp32_dtype)
        unet_input = torch.cat([rgb_latent, init_latent], dim=1)
        _log_dt("vae.encode left/init + cat 8ch", t4)

        if self.unet.config.in_channels != unet_input.shape[1]:
            raise ValueError(
                f"UNet expects {self.unet.config.in_channels} input channels, "
                f"but pipeline built {unet_input.shape[1]}. 请确认加载的是训练后 8-channel UNet checkpoint。"
            )

        # -----------------------------
        # F) RGB ControlNet input = warped right latent, same as training
        # -----------------------------
        t5 = _cuda_time()
        if self.mono:
            right_warpm11 = left_m11
        else:
            right_warpm11 = warp_right_to_left(
                right_m11,
                disp_pred_safe.detach(),
                sign=warp_sign,
                align_corners=True,
            ).to(dtype=left_m11.dtype)
        with no_autocast:
            warp_to_left_latent = (
                self.vae.encode(right_warpm11.to(dtype=vae_dtype)).latent_dist.mode()
                * self.vae.config.scaling_factor
            )
        warp_to_left_latent = warp_to_left_latent.to(device=device, dtype=fp32_dtype)
        _log_dt("warp right + vae.encode", t5)

        # -----------------------------
        # G) prompt embeddings
        # -----------------------------
        t6 = _cuda_time()
        if prompt_embeds is None:
            if self.prompt_embeds is None or prompt is not None:
                prompt_str = prompt if prompt is not None else getattr(self, "prompt", "a normal map")
                text_inputs = self.tokenizer(
                    prompt_str,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                self.prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))[0]
            prompt_embeds_base = self.prompt_embeds
        else:
            prompt_embeds_base = prompt_embeds

        prompt_embeds_fp32 = prompt_embeds_base.repeat(B, 1, 1).to(device=device, dtype=torch.float32)
        prompt_embeds_mp = prompt_embeds_fp32.to(device=device, dtype=fp32_dtype)
        _log_dt("text encoder / prompt embeds", t6)

        # Fixed training timestep.
        timesteps = torch.full((B,), t_fixed, device=device, dtype=torch.long)

        # -----------------------------
        # H) DINO ControlNet, using FoundationStereo left_tokens
        # -----------------------------
        t7 = _cuda_time()
        with no_autocast:
            if left_tokens is None:
                # Fallback only. Training uses left_tokens returned by FoundationStereo.
                dino_out = self.FoundationStereo.feature.dino(
                    left_stereo.float(), return_tokens=True, return_depth=False
                )
                left_tokens = dino_out["tokens"] if isinstance(dino_out, dict) else dino_out

            dino_dtype = fp32_dtype
            dino_device = device
            dino_tokens_raw = left_tokens
            dino_tokens_raw = dino_tokens_raw.to(device=dino_device, dtype=dino_dtype)
            dino_feat = self.dino_controlnet.dino_controlnet_cond_embedding(dino_tokens_raw)
            if dino_feat.shape[-2:] != unet_input.shape[-2:]:
                dino_feat = F.interpolate(
                    dino_feat,
                    size=unet_input.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            dino_feat = dino_feat.to(device=dino_device, dtype=dino_dtype)    
            dino_dtype = fp32_dtype
            dino_device = device   
            prompt_embeds_dino = prompt_embeds_fp32.to(
                device=dino_device,
                dtype=dino_dtype,
            ) 
            dino_down_res, _ = self.dino_controlnet(
                dino_feat,
                timesteps.to(dino_device),
                encoder_hidden_states=prompt_embeds_dino,
                conditioning_scale=controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )
        dino_down_res_copy = [r.to(device=device, dtype=fp32_dtype) for r in dino_down_res]
        _log_dt("dino_controlnet", t7)

        # -----------------------------
        # I) RGB ControlNet + one-shot UNet latent prediction
        # -----------------------------
        t8 = _cuda_time()
        down_res, mid_res = self.controlnet(
            warp_to_left_latent.detach(),
            timesteps,
            encoder_hidden_states=prompt_embeds_mp,
            conditioning_scale=controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )
        down_res = [r.to(device=device, dtype=fp32_dtype) for r in down_res]
        mid_res = mid_res.to(device=device, dtype=fp32_dtype)

        task_emb = torch.tensor([1, 0], device=device, dtype=torch.float32).unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        task_emb = task_emb.repeat(B, 1).to(device=device, dtype=fp32_dtype)

        pred_latent_x0 = self.dino_unet_forward(
            self.unet,
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds_mp,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            dino_down_block_additional_residuals=dino_down_res_copy,
            class_labels=task_emb,
            return_dict=False,
        )[0]
        _log_dt("controlnet + one-shot unet", t8)

        # -----------------------------
        # J) decode predicted normal latent
        # -----------------------------
        t9 = _cuda_time()
        prediction = self.decode_prediction(pred_latent_x0)
        prediction = self.normalize_normals(prediction.float()).to(dtype=fp32_dtype)
        _log_dt("vae.decode + normalize", t9)

        # -----------------------------
        # K) resize prediction, disparity, and init normal back to original size
        # -----------------------------
        t10 = _cuda_time()
        if match_input_resolution and prediction.shape[-2:] != original_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )
            prediction = self.normalize_normals(prediction.float()).to(dtype=fp32_dtype)

        disp_out = disp_pred_safe
        normal_init_out = normal_init
        if match_input_resolution and disp_pred_safe.shape[-2:] != original_resolution:
            disp_out = F.interpolate(disp_pred_safe, size=original_resolution, mode="bilinear", align_corners=False)
            disp_scale = float(original_resolution[1]) / float(W_resize)
            disp_out = disp_out * disp_scale
            normal_init_out = F.interpolate(normal_init, size=original_resolution, mode="bilinear", align_corners=False)
            normal_init_out = normalize_normals(normal_init_out.float()).to(dtype=fp32_dtype)
        _log_dt("resize outputs", t10)

        # -----------------------------
        # L) output convert
        # -----------------------------
        t11 = _cuda_time()
        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)
            prediction = prediction.clip(min=-1, max=1)
        elif output_type == "pt":
            prediction = prediction.clamp(-1, 1)
        else:
            raise ValueError(f"Unsupported output_type={output_type}. Use 'np' or 'pt'.")
        _log_dt("output convert", t11)

        _log_dt("TOTAL __call__", t_all)

        return StableNormalOutput(
            prediction=prediction,
            latent=pred_latent_x0,
            gaus_noise=None,
            disp_pred=disp_out,
            normal_init=normal_init_out,
        )


    # Copied from diffusers.pipelines.marigold.pipeline_marigold_depth.MarigoldDepthPipeline.prepare_latents
    def prepare_latents(
        self,
        image: torch.Tensor,
        latents: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]

        pred_latent = latents
        if pred_latent is None:
            pred_latent = randn_tensor(
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        return prediction  # [B,3,H,W]

    @staticmethod
    def normalize_normals(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")

        norm = torch.norm(normals, dim=1, keepdim=True)
        normals /= norm.clamp(min=eps)

        return normals

    @staticmethod
    def match_noisy(dino, noisy):
        _, __, dino_h, dino_w =  dino.shape
        _, __, h, w =  noisy.shape

        if h == dino_h and w == dino_w:
            return dino
        else:
            return F.interpolate(dino, (h, w), mode='bilinear')

    @staticmethod
    def dino_unet_forward(
        self,  # NOTE that repurpose to UNet
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        dino_down_block_additional_residuals: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        
        default_overall_up_factor = 2**self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        sample = self.conv_in(sample)

        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None

        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        def residual_downforward(
            self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None,
            additional_residuals: Optional[torch.Tensor] = None,
            *args, **kwargs,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            
            output_states = ()

            for resnet in self.resnets:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states


        def residual_blockforward(
            self, 
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            additional_residuals: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            output_states = ()

            blocks = list(zip(self.resnets, self.attentions))

            for i, (resnet, attn) in enumerate(blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                hidden_states += additional_residuals.pop(0)
                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states


        down_intrablock_additional_residuals = dino_down_block_additional_residuals

        sample += down_intrablock_additional_residuals.pop(0)
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = residual_blockforward(
                    downsample_block,
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    additional_residuals = down_intrablock_additional_residuals,
                )
            else:
                sample, res_samples = residual_downforward(
                    downsample_block,
                    hidden_states=sample,
                    temb=emb,
                    additional_residuals = down_intrablock_additional_residuals,
                        )

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


    @staticmethod
    def ensemble_normals(
        normals: torch.Tensor, output_uncertainty: bool, reduction: str = "closest"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")
        if reduction not in ("closest", "mean"):
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
        mean_normals = MarigoldNormalsPipeline.normalize_normals(mean_normals)  # [1,3,H,W]

        sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]
        sim_cos = sim_cos.clamp(-1, 1)  

        uncertainty = None
        if output_uncertainty:
            uncertainty = sim_cos.arccos()  # [E,1,H,W]
            uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

        if reduction == "mean":
            return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

        closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
        closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
        closest_normals = torch.gather(normals, 0, closest_indices)  # [1,3,H,W]

        return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
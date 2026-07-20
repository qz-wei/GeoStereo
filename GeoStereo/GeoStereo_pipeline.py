from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect
import json
import os,sys
import numpy as np
import torch
from safetensors.torch import load_file
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
from diffusers.schedulers import DDPMScheduler

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
from .core.foundation_stereo import FoundationStereo
from .warp_controlnetvae import Warp_ControlNetVAEModel
from .normal_controlnetvae import Normal_ControlNetVAEModel
from huggingface_hub import snapshot_download
from einops import rearrange  
import torchvision
import time

logger = logging.get_logger(__name__)  

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DINO_PATCH = 14  

def normalize_normals(n: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return n / (torch.norm(n, dim=1, keepdim=True) + eps)

def warp_right_to_left(
    right_bchw: torch.Tensor,
    disp_b1hw: torch.Tensor,
    sign: str = "minus",  
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

def normalize_image_imagenet(img: torch.Tensor) -> torch.Tensor:
    img01 = img_to_01(img)
    tf = torchvision.transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD), inplace=False)
    return tf(img01).contiguous()




@dataclass
class GeoStereoOutput(BaseOutput):
    """GeoStereo inference result.

    Attributes:
        normal: Predicted surface normal map.
        disparity: Predicted disparity map.
    """

    normal: Union[np.ndarray, torch.Tensor]
    disparity: torch.Tensor

class GeoStereoPipeline(StableDiffusionControlNetPipeline):
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
        normal_controlnet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker=True,
        default_denoising_steps=1,
        default_processing_resolution=768,
        prompt="",
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
            normal_controlnet=normal_controlnet,
        )
        self.args = args
        self.image_processor = MarigoldImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

        foundation_config = dict(args.foundation_config)
        foundation_config.update(
            {
                key: value
                for key, value in vars(args).items()
                if key != "foundation_config"
            }
        )
        self.margs = OmegaConf.create(foundation_config)
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.prompt = prompt
        self.prompt_embeds = None
        self.empty_text_embedding = empty_text_embedding
        depth_any = DepthAnythingFeature(
            encoder=str(getattr(self.margs, "vit_size", "vitl")),
            tokens_out_dim=None,
            resize_to=None,
        )
        self.mono = args.mono
        self.FoundationStereo = FoundationStereo(self.margs, depth_any)
        self.train_iters = int(getattr(args, "train_iters", 10))
        #self.register_modules(args=args)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, os.PathLike],
        *,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.float32,
        enable_xformers: bool = False,
        **overrides,
    ) -> "GeoStereoPipeline":
        """Load GeoStereo from a fixed local or Hugging Face model layout."""
        candidate = Path(
            os.path.expandvars(
                os.path.expanduser(str(model_path))
            )
        )

        root = (
            candidate.resolve()
            if candidate.is_dir()
            else Path(
                snapshot_download(repo_id=str(model_path))
            ).resolve()
        )

        geostereo_config_path = root / "GeoStereo_config.json"
        foundation_config_path = (
            root / "foundation_stereo" / "config.json"
        )
        foundation_weights_path = (
            root
            / "foundation_stereo"
            / "model.safetensors"
        )

        required_paths = (
            geostereo_config_path,
            root / "unet",
            root / "controlnet",
            root / "normal_controlnet",
            root / "vae",
            root / "text_encoder",
            root / "tokenizer",
            root / "scheduler",
            foundation_config_path,
            foundation_weights_path,
        )
        missing_paths = [
            str(path.relative_to(root))
            for path in required_paths
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"GeoStereo model directory is incomplete: {root}. "
                f"Missing: {missing_paths}"
            )

        with geostereo_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            release_config: Dict[str, Any] = json.load(file)

        runtime_config: Dict[str, Any] = {
            "mono": False,
            "train_iters": 15,
            "max_disp": 416,
            "timestep": 999,
            "warp_sign": "minus",
            "warp_control_scale": 0.2,
            "normal_control_scale": 0.2,
            "infer_divider": 32,
            "infer_max_res": 2240,
            "vae_latent_sampling": "sample",
        }
        runtime_config.update(release_config)
        runtime_config.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )

        with foundation_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            foundation_config: Dict[str, Any] = json.load(file)

        pipe_args = SimpleNamespace(
            foundation_config=foundation_config,
            mono=bool(runtime_config["mono"]),
            train_iters=int(runtime_config["train_iters"]),
            max_disp=int(runtime_config["max_disp"]),
            timestep=int(runtime_config["timestep"]),
            warp_sign=str(runtime_config["warp_sign"]),
            warp_control_scale=float(
                runtime_config["warp_control_scale"]
            ),
            normal_control_scale=float(
                runtime_config["normal_control_scale"]
            ),
            infer_divider=int(runtime_config["infer_divider"]),
            infer_max_res=int(runtime_config["infer_max_res"]),
            vae_latent_sampling=str(
                runtime_config["vae_latent_sampling"]
            ),
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            str(root),
            subfolder="tokenizer",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            str(root),
            subfolder="text_encoder",
        )
        vae = AutoencoderKL.from_pretrained(
            str(root),
            subfolder="vae",
        )
        scheduler = DDPMScheduler.from_pretrained(
            str(root),
            subfolder="scheduler",
        )
        unet = UNet2DConditionModel.from_pretrained(
            str(root),
            subfolder="unet",
        )
        warp_controlnet = Warp_ControlNetVAEModel.from_pretrained(
            str(root),
            subfolder="controlnet",
        )
        normal_controlnet = Normal_ControlNetVAEModel.from_pretrained(
            str(root),
            subfolder="normal_controlnet",
        )

        pipe = cls(
            args=pipe_args,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=warp_controlnet,
            normal_controlnet=normal_controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
            default_denoising_steps=1,
            default_processing_resolution=0,
            prompt="",
            empty_text_embedding=None,
        )

        foundation_state_dict = load_file(
            str(foundation_weights_path),
            device="cpu",
        )
        missing_keys, unexpected_keys = (
            pipe.FoundationStereo.load_state_dict(
                foundation_state_dict,
                strict=False,
            )
        )

        if missing_keys:
            logger.warning(
                "FoundationStereo missing keys (%d): %s",
                len(missing_keys),
                missing_keys,
            )
        if unexpected_keys:
            logger.warning(
                "FoundationStereo unexpected keys (%d): %s",
                len(unexpected_keys),
                unexpected_keys,
            )

        device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        pipe.set_inference_dtype(
            device=device,
            torch_dtype=torch_dtype,
        )

        if enable_xformers:
            pipe.unet.enable_xformers_memory_efficient_attention()

        pipe.eval()
        pipe.model_root = root
        pipe.release_config = runtime_config
        return pipe

    def set_inference_dtype(
        self,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """Move diffusion modules to the selected dtype and keep FoundationStereo in fp32."""
        if device is None:
            device = self._execution_device
        device = torch.device(device)

        for name in (
            "vae",
            "text_encoder",
            "unet",
            "controlnet",
            "normal_controlnet",
        ):
            module = getattr(self, name, None)
            if module is not None:
                module.to(device=device, dtype=torch_dtype)
                module.eval()

        self.FoundationStereo = self.FoundationStereo.to(
            device=device,
            dtype=torch.float32,
        )
        self.FoundationStereo.eval()
        try:
            self.FoundationStereo.feature = self.FoundationStereo.feature.to(
                device=device,
                dtype=torch.float32,
            )
            self.FoundationStereo.feature.dino = self.FoundationStereo.feature.dino.to(
                device=device,
                dtype=torch.float32,
            )
        except Exception:
            pass

        return self

    def eval(self):
        for name in (
            "vae",
            "text_encoder",
            "unet",
            "controlnet",
            "normal_controlnet",
            "FoundationStereo",
        ):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "eval"):
                module.eval()
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
        prompt: Optional[str] = None,
        match_input_resolution: bool = True,
        resample_method_output: str = "bilinear",
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        warp_controlnet_conditioning_scale: Optional[float] = None,
        normal_controlnet_conditioning_scale: Optional[float] = None,
        output_type: str = "np",
        return_dict: bool = True,
        **kwargs,
    ):
        """
        One-shot inference aligned with the released training objective.

        Training input:
            UNet input = [VAE(left RGB), noisy target-normal latent]
            Warp ControlNet input = VAE(warped right RGB), timestep=t
            Normal ControlNet input = VAE(stereo normal prior), timestep=0
            UNet target = clean target-normal latent x0

        Since the target normal is unknown at inference time, the second UNet
        input is sampled Gaussian noise at the fixed training timestep.

        Returns only the final surface normal and FoundationStereo disparity.
        The intermediate stereo-derived normal prior is used internally but is
        not exposed in the public output.
        """
        if not torch.is_tensor(image1) or not torch.is_tensor(image2):
            raise TypeError(
                "`image1` and `image2` must be torch tensors. "
                "Use infer.py for loading image paths."
            )

        device = self._execution_device
        model_dtype = next(self.unet.parameters()).dtype
        vae_dtype = next(self.vae.parameters()).dtype

        timestep_value = int(getattr(self.args, "timestep", 999))
        max_disp = float(getattr(self.args, "max_disp", 416))
        train_iters = int(getattr(self.args, "train_iters", 15))
        warp_sign = str(getattr(self.args, "warp_sign", "minus"))
        vae_sampling = str(
            getattr(self.args, "vae_latent_sampling", "mode")
        ).lower()

        legacy_scale = controlnet_conditioning_scale
        if warp_controlnet_conditioning_scale is None:
            warp_controlnet_conditioning_scale = (
                float(legacy_scale)
                if legacy_scale is not None
                else float(getattr(self.args, "warp_control_scale", 0.2))
            )
        if normal_controlnet_conditioning_scale is None:
            normal_controlnet_conditioning_scale = (
                float(legacy_scale)
                if legacy_scale is not None
                else float(getattr(self.args, "normal_control_scale", 0.2))
            )

        left_m11 = img_to_m11(image1).to(
            device=device,
            dtype=model_dtype,
            non_blocking=True,
        )
        right_m11 = img_to_m11(image2).to(
            device=device,
            dtype=model_dtype,
            non_blocking=True,
        )

        if left_m11.shape != right_m11.shape:
            raise ValueError(
                "Left/right tensors must have the same shape, got "
                f"{tuple(left_m11.shape)} and {tuple(right_m11.shape)}."
            )

        _, _, original_h, original_w = left_m11.shape
        original_resolution = (original_h, original_w)

        divider = int(getattr(self.args, "infer_divider", 32))
        max_res = int(getattr(self.args, "infer_max_res", 2240))
        resize_h, resize_w = get_resize_keep_aspect_ratio(
            original_h,
            original_w,
            divider=divider,
            max_H=max_res,
            max_W=max_res,
        )

        if (resize_h, resize_w) != original_resolution:
            left_m11 = F.interpolate(
                left_m11,
                size=(resize_h, resize_w),
                mode="bicubic",
                align_corners=False,
            )
            right_m11 = F.interpolate(
                right_m11,
                size=(resize_h, resize_w),
                mode="bicubic",
                align_corners=False,
            )

        batch_size = left_m11.shape[0]
        no_autocast = (
            torch.autocast("cuda", enabled=False)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

        left_stereo = normalize_image_imagenet(left_m11).float()
        right_stereo = normalize_image_imagenet(right_m11).float()

        with no_autocast:
            _, disp_predictions, _ = self.FoundationStereo(
                left_stereo,
                right_stereo,
                iters=train_iters,
                test_mode=False,
            )
            disp_pred = disp_predictions[-1].float()
            if disp_pred.ndim == 3:
                disp_pred = disp_pred.unsqueeze(1)
            disp_pred_safe = disp_pred.clamp(
                min=1e-3,
                max=max_disp,
            )

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
            normal_init = (
                normalize_normals(normal_init) * valid_normal
            ).clamp(-1.0, 1.0)

        if bool(getattr(self.args, "mono", False)):
            right_warp_m11 = left_m11
        else:
            right_warp_m11 = warp_right_to_left(
                right_m11,
                disp_pred_safe.detach().to(dtype=right_m11.dtype),
                sign=warp_sign,
                align_corners=True,
            )

        def encode_vae(image: torch.Tensor) -> torch.Tensor:
            posterior = self.vae.encode(
                image.to(device=device, dtype=vae_dtype)
            ).latent_dist
            if vae_sampling == "sample":
                try:
                    latent = posterior.sample(generator=generator)
                except TypeError:
                    latent = posterior.sample()
            elif vae_sampling == "mode":
                latent = posterior.mode()
            else:
                raise ValueError(
                    "vae_latent_sampling must be 'sample' or 'mode', "
                    f"got {vae_sampling!r}."
                )
            return latent * self.vae.config.scaling_factor

        with no_autocast:
            rgb_latent = encode_vae(left_m11)
            init_latent = encode_vae(normal_init)
            warp_latent = encode_vae(right_warp_m11)

        rgb_latent = rgb_latent.to(device=device, dtype=model_dtype)
        init_latent = init_latent.to(device=device, dtype=model_dtype)
        warp_latent = warp_latent.to(device=device, dtype=model_dtype)

        timesteps = torch.full(
            (batch_size,),
            timestep_value,
            device=device,
            dtype=torch.long,
        )

        if latents is None:
            gaussian_noise = randn_tensor(
                init_latent.shape,
                generator=generator,
                device=device,
                dtype=model_dtype,
            )
        else:
            gaussian_noise = latents.to(device=device, dtype=model_dtype)
            if gaussian_noise.shape != init_latent.shape:
                raise ValueError(
                    "`latents` must match the normal latent shape "
                    f"{tuple(init_latent.shape)}, got {tuple(gaussian_noise.shape)}."
                )

        # Match the noise amplitude used by scheduler.add_noise at training.
        noisy_target_latent = self.scheduler.add_noise(
            torch.zeros_like(gaussian_noise),
            gaussian_noise,
            timesteps,
        )
        unet_input = torch.cat(
            [rgb_latent, noisy_target_latent],
            dim=1,
        )

        if int(self.unet.config.in_channels) != 8:
            raise RuntimeError(
                "GeoStereo requires the trained 8-channel UNet, but the loaded "
                f"UNet reports in_channels={self.unet.config.in_channels}."
            )

        if prompt_embeds is None:
            if self.prompt_embeds is None or prompt is not None:
                prompt_text = "" if prompt is None else prompt
                text_inputs = self.tokenizer(
                    prompt_text,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                self.prompt_embeds = self.text_encoder(
                    text_inputs.input_ids.to(device)
                )[0]
            prompt_embeds_base = self.prompt_embeds
        else:
            prompt_embeds_base = prompt_embeds

        prompt_embeds_model = prompt_embeds_base.repeat(
            batch_size, 1, 1
        ).to(device=device, dtype=model_dtype)

        normal_timesteps = torch.zeros_like(timesteps)
        normal_down_residuals, _ = self.normal_controlnet(
            sample=init_latent.detach(),
            timestep=normal_timesteps,
            encoder_hidden_states=prompt_embeds_model,
            conditioning_scale=float(
                normal_controlnet_conditioning_scale
            ),
            guess_mode=False,
            return_dict=False,
        )
        normal_down_residuals = [
            residual.to(device=device, dtype=model_dtype)
            for residual in normal_down_residuals
        ]

        warp_down_residuals, warp_mid_residual = self.controlnet(
            sample=warp_latent.detach(),
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds_model,
            conditioning_scale=float(
                warp_controlnet_conditioning_scale
            ),
            guess_mode=False,
            return_dict=False,
        )
        warp_down_residuals = [
            residual.to(device=device, dtype=model_dtype)
            for residual in warp_down_residuals
        ]
        warp_mid_residual = warp_mid_residual.to(
            device=device,
            dtype=model_dtype,
        )

        task_embedding = torch.tensor(
            [1.0, 0.0],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        task_embedding = torch.cat(
            [torch.sin(task_embedding), torch.cos(task_embedding)],
            dim=-1,
        )
        task_embedding = task_embedding.repeat(
            batch_size, 1
        ).to(dtype=model_dtype)

        pred_latent_x0 = self.dino_unet_forward(
            self.unet,
            unet_input,
            timesteps,
            encoder_hidden_states=prompt_embeds_model,
            down_block_additional_residuals=warp_down_residuals,
            mid_block_additional_residual=warp_mid_residual,
            dino_down_block_additional_residuals=normal_down_residuals,
            class_labels=task_embedding,
            return_dict=False,
        )[0]

        prediction = self.decode_prediction(pred_latent_x0)
        prediction = self.normalize_normals(
            prediction.float()
        )

        if (
            match_input_resolution
            and prediction.shape[-2:] != original_resolution
        ):
            prediction = self.image_processor.resize_antialias(
                prediction,
                original_resolution,
                resample_method_output,
                is_aa=False,
            )
            prediction = self.normalize_normals(
                prediction.float()
            )

        disparity_output = disp_pred_safe
        if (
            match_input_resolution
            and disp_pred_safe.shape[-2:] != original_resolution
        ):
            disparity_output = F.interpolate(
                disp_pred_safe,
                size=original_resolution,
                mode="bilinear",
                align_corners=False,
            )
            disparity_output = disparity_output * (
                float(original_w) / float(resize_w)
            )

        if output_type == "np":
            normal_output = self.image_processor.pt_to_numpy(
                prediction
            ).clip(min=-1, max=1)
        elif output_type == "pt":
            normal_output = prediction.clamp(-1, 1)
        else:
            raise ValueError(
                f"Unsupported output_type={output_type!r}; use 'np' or 'pt'."
            )

        output = GeoStereoOutput(
            normal=normal_output,
            disparity=disparity_output,
        )
        if return_dict:
            return output
        return output.normal, output.disparity


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

# Backward-compatible aliases.
StableNormalPipeline = GeoStereoPipeline
StableNormalOutput = GeoStereoOutput
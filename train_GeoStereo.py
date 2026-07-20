import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
import accelerate
import datasets
import torch
from safetensors.torch import load_file
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import diffusers
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from GeoStereo.normal_controlnetvae import Normal_ControlNetVAEModel
from GeoStereo.depthanything_head import DepthAnythingFeature
from util.utils import (
    dino_unet_forward,
    disp_to_normal_points_no_intrinsics,
    set_requires_grad,
    unwrap,
    warp_right_to_left,
)
import torchvision
from dataloaders.load_stereo import HypersimStereoJSONL,IRSStereo,VirtualKITTI2Stereo,MixedDataLoaderN
from util.lr_scheduler import IterExponential
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler

from GeoStereo.warp_controlnetvae import Warp_ControlNetVAEModel

from GeoStereo.core.foundation_stereo import FoundationStereo
import time

if is_wandb_available():
    import wandb

class GeoUNetWrapper(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
    ):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_additional_residuals,
        mid_block_additional_residual,
        dino_down_block_additional_residuals,
        class_labels,
    ):
        return dino_unet_forward(
            self.unet,
            sample,
            timestep,
            encoder_hidden_states=(
                encoder_hidden_states
            ),
            down_block_additional_residuals=(
                down_block_additional_residuals
            ),
            mid_block_additional_residual=(
                mid_block_additional_residual
            ),
            dino_down_block_additional_residuals=(
                dino_down_block_additional_residuals
            ),
            class_labels=class_labels,
            return_dict=False,
        )[0]


check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def stable_cosine_normal_loss(
    pred_normal: torch.Tensor,
    gt_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:

    pred_normal = F.normalize(
        pred_normal.float(),
        p=2,
        dim=1,
        eps=eps,
    )

    gt_normal = F.normalize(
        gt_normal.float(),
        p=2,
        dim=1,
        eps=eps,
    )

    cosine = torch.sum(
        pred_normal * gt_normal,
        dim=1,
        keepdim=True,
    )

    # 不使用 acos，同时避开数值边界
    cosine = cosine.clamp(
        min=-1.0 + eps,
        max=1.0 - eps,
    )

    loss_map = 1.0 - cosine

    valid_mask = mask_to_bool(
        valid_mask,
        size_hw=loss_map.shape[-2:],
    ).float()

    return (
        loss_map * valid_mask
    ).sum() / (valid_mask.sum() + eps)

def encode_vae_latent(vae, image, sampling="sample"):
    """
    Encode a [-1, 1] image tensor into a scaled Stable-Diffusion latent.

    sampling="sample" uses stochastic posterior sampling.
    sampling="mode" gives deterministic posterior means.
    """
    posterior = vae.encode(image).latent_dist
    if sampling == "sample":
        latent = posterior.sample()
    elif sampling == "mode":
        latent = posterior.mode()
    else:
        raise ValueError(f"Unknown VAE latent sampling mode: {sampling}")

    return latent * vae.config.scaling_factor

def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoStereo normal training from a complete GeoStereo checkpoint."
    )

    # Complete initialization checkpoint. Every model component is loaded
    # from this one directory.
    parser.add_argument(
        "--restore_ckpt",
        type=str,
        required=True,
        help=(
            "GeoStereo model directory containing unet, controlnet, "
            "normal_controlnet, vae, text_encoder, tokenizer, scheduler, "
            "and foundation_stereo."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="outputs/ckpt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int,  default=40000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Optimizer / LR: keep first-script style
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=40000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Dataloader / logging / distributed
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="outputs_718/logs")
    parser.add_argument("--mixed_precision", type=str, default='no', choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention",default=True, action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default="e2e-ft-diffusion")
    parser.add_argument("--mono", type=bool, default=False)
    parser.add_argument("--warp_sign", type=str, default="minus", choices=["minus", "plus"],
                   help="warp sign: minus means x_r = x_l - disp (common).")
    # Training-time controls used inside the current epoch logic
    #parser.add_argument("--train_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=416)
    parser.add_argument("--control_scale", type=float, default=0.2,
                    help="Backward-compatible default used when a branch-specific scale is omitted.")
    parser.add_argument("--warp_control_scale", type=float, default=None,
                    help="Conditioning scale for the warp-to-left-view ControlNet.")
    parser.add_argument("--normal_control_scale", type=float, default=None,
                    help="Conditioning scale for the FoundationStereo-normal ControlNet.")
    parser.add_argument("--lambda_ang", type=float, default=0.2)
    parser.add_argument("--vis_step", type=int, default=5)

    # Fixed-timestep x0 prediction
    parser.add_argument("--timestep", type=int, default=999,
                        help="Fixed diffusion timestep used to corrupt target-normal latents.")
    parser.add_argument("--prediction_type", type=str, default="sample",
                        choices=["sample"],
                        help="Predict the clean target-normal latent x0.")
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--input_perturbation", type=float, default=0.0)
    parser.add_argument("--vae_latent_sampling", type=str, default="sample",
                        choices=["sample", "mode"],
                        help="Use sample for stochastic VAE encoding; mode is deterministic.")
    parser.add_argument("--control_extra_scale", type=float, default=1e-3,
                        help="Deprecated; retained only for command-line compatibility.")
    

    args = parser.parse_args()
    if args.warp_control_scale is None:
        args.warp_control_scale = args.control_scale
    if args.normal_control_scale is None:
        args.normal_control_scale = args.control_scale
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def safe_vis(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach(), nan=0.0, posinf=1.0, neginf=0.0)

def norm_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

def to_grid01(x: torch.Tensor, nrow: int = 1) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    grid = torchvision.utils.make_grid(x.clamp(0.0, 1.0), nrow=nrow)
    return grid

def scalar01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.detach().float()
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)
def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape={tuple(x.shape)}")
    return x.contiguous().float()


def m11_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def normalize_image_imagenet(x01: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x01.device, dtype=x01.dtype)
    std = IMAGENET_STD.to(device=x01.device, dtype=x01.dtype)
    return (x01 - mean) / std


def normalize_normals(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps)


def mask_to_bool(mask: torch.Tensor, size_hw=None) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    if size_hw is not None and tuple(mask.shape[-2:]) != tuple(size_hw):
        mask = F.interpolate(mask, size=size_hw, mode="nearest")
    return mask > 0.5

@torch.no_grad()
def build_prompt_embeds(tokenizer, text_encoder, prompt: str, device, dtype=torch.float32):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    return prompt_embeds.to(dtype=dtype)
def main():
    args = parse_args()

    logging_dir = args.logging_dir
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    amp_device_type = device.type

    if accelerator.mixed_precision == "fp16":
        mp_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        mp_dtype = torch.bfloat16
    else:
        mp_dtype = torch.float32

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, "w") as file:
            file.write(args_str)

    # ---------------------------------------------------------------------
    # Load every component from one complete GeoStereo checkpoint.
    #
    # Expected layout:
    #   restore_ckpt/
    #   ├── unet/
    #   ├── controlnet/
    #   ├── normal_controlnet/
    #   ├── vae/
    #   ├── text_encoder/
    #   ├── tokenizer/
    #   ├── scheduler/
    #   └── foundation_stereo/
    #       ├── config.json
    #       └── model.safetensors
    # ---------------------------------------------------------------------
    restore_ckpt = Path(args.restore_ckpt).expanduser().resolve()

    required_paths = (
        restore_ckpt / "unet",
        restore_ckpt / "controlnet",
        restore_ckpt / "normal_controlnet",
        restore_ckpt / "vae",
        restore_ckpt / "text_encoder",
        restore_ckpt / "tokenizer",
        restore_ckpt / "scheduler",
        restore_ckpt / "foundation_stereo" / "config.json",
        restore_ckpt / "foundation_stereo" / "model.safetensors",
    )
    missing_paths = [
        str(path.relative_to(restore_ckpt))
        for path in required_paths
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Incomplete restore checkpoint: {restore_ckpt}. "
            f"Missing: {missing_paths}"
        )

    logger.info(f"Loading all model components from {restore_ckpt}")

    tokenizer = CLIPTokenizer.from_pretrained(
        str(restore_ckpt),
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(restore_ckpt),
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        str(restore_ckpt),
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(restore_ckpt),
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    rgb_ctl = Warp_ControlNetVAEModel.from_pretrained(
        str(restore_ckpt),
        subfolder="controlnet",
    )
    normal_ctl = Normal_ControlNetVAEModel.from_pretrained(
        str(restore_ckpt),
        subfolder="normal_controlnet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        str(restore_ckpt),
        subfolder="scheduler",
    )
    noise_scheduler.register_to_config(
        prediction_type=args.prediction_type
    )

    if int(unet.config.in_channels) != 8:
        raise RuntimeError(
            "The restored GeoStereo UNet must have 8 input channels, "
            f"got {unet.config.in_channels}."
        )
    if int(rgb_ctl.conv_in.in_channels) != 4:
        raise RuntimeError(
            "The restored warp ControlNet must have 4 input channels, "
            f"got {rgb_ctl.conv_in.in_channels}."
        )

    normal_channels = int(
        getattr(
            normal_ctl.config,
            "dino_conditioning_channels",
            -1,
        )
    )
    if normal_channels != 4:
        raise RuntimeError(
            "The restored normal ControlNet must accept a 4-channel normal "
            f"latent, got {normal_channels}."
        )

    foundation_dir = restore_ckpt / "foundation_stereo"
    with (foundation_dir / "config.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        foundation_config = json.load(file)

    foundation_config.update(
        {
            "max_disp": int(args.max_disp),
            "mixed_precision": False,
        }
    )
    foundation_args = OmegaConf.create(foundation_config)

    depth_any = DepthAnythingFeature(
        encoder=str(
            foundation_config.get("vit_size", "vitl")
        ),
        tokens_out_dim=None,
        resize_to=None,
    )
    foundation = FoundationStereo(
        foundation_args,
        depth_any,
    )

    foundation_state_dict = load_file(
        str(foundation_dir / "model.safetensors"),
        device="cpu",
    )
    missing_keys, unexpected_keys = foundation.load_state_dict(
        foundation_state_dict,
        strict=False,
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

    foundation.to(
        device=device,
        dtype=torch.float32,
    )
    foundation.requires_grad_(False)
    foundation.eval()

    text_encoder.to(device)
    if rgb_ctl is None:
        raise ValueError("StableNormalPipeline.controlnet is None, but this script is configured to train rgb_ctl.")

    if unet.config["in_channels"] != 8:
        raise RuntimeError(
            f"GeoStereo UNet must have 8 input channels, got {unet.config['in_channels']}."
        )
    logger.info(
        "UNet input: [RGB latent, noisy target-normal latent] (8 channels), "
        "prediction target: clean target-normal latent x0."
    )
    logger.info(
        "Warp ControlNet input: warp-to-left VAE latent "
        f"({rgb_ctl.conv_in.in_channels} channels)."
    )
    logger.info(
        "Normal ControlNet input: FoundationStereo normal VAE latent "
        f"({normal_ctl.config.dino_conditioning_channels} channels)."
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 may be unstable for training on some GPUs. Prefer >= 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Freeze non-trainable modules
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    foundation.requires_grad_(False).eval()

    # Trainable modules
    unet.train(True)
    set_requires_grad(unet, True)

    normal_ctl.train(True)
    set_requires_grad(normal_ctl, True)

    rgb_ctl.train(True)
    set_requires_grad(rgb_ctl, True)

    with torch.no_grad():
        prompt_embeds_1 = build_prompt_embeds(tokenizer=tokenizer, text_encoder=text_encoder,prompt='',device=device, dtype=torch.float32)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model



    # Do not register custom Accelerate save/load hooks here.
    # GeoUNetWrapper and the two ControlNets are passed directly to
    # accelerator.prepare(), so accelerator.save_state() can save their
    # distributed training states automatically. Model-only Diffusers
    # checkpoints are saved explicitly below.

    geo_unet = GeoUNetWrapper(unet)
    optimizer = torch.optim.AdamW(
        [
            {"params": geo_unet.parameters()},
            {"params": normal_ctl.parameters()},
            {"params": rgb_ctl.parameters()},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # lr_func = IterExponential(
    #     total_iter_length=args.lr_total_iter_length ,
    #     final_ratio=0.01,
    #     warmup_steps=args.lr_exp_warmup_steps,
    # )
    # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    scheduler_process_scale = (
        1
        if accelerator.split_batches
        else accelerator.num_processes
    )

    scaled_warmup_steps = (
        args.lr_exp_warmup_steps
        * scheduler_process_scale
    )

    scaled_total_steps = (
        args.lr_total_iter_length
        * scheduler_process_scale
    )

    lr_func = IterExponential(
        total_iter_length=scaled_total_steps,
        final_ratio=0.01,
        warmup_steps=scaled_warmup_steps,
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_func,
    )

    logger.info(
        f"LR scheduler: global warmup={args.lr_exp_warmup_steps}, "
        f"global total={args.lr_total_iter_length}, "
        f"process scale={scheduler_process_scale}, "
        f"internal warmup={scaled_warmup_steps}, "
        f"internal total={scaled_total_steps}"
    )


    train_dataset_hypersim = HypersimStereoJSONL(transform=True)
    train_dataset_vkitti = VirtualKITTI2Stereo( transform=True)
    train_dataset_irs = IRSStereo(transform = True)
    train_dataloader_irs = torch.utils.data.DataLoader(
        train_dataset_irs,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    train_dataloader_vkitti = torch.utils.data.DataLoader(
        train_dataset_vkitti,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    train_dataloader_hypersim = torch.utils.data.DataLoader(
        train_dataset_hypersim,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    geo_unet, normal_ctl, rgb_ctl, optimizer, \
        train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_irs, \
        lr_scheduler = accelerator.prepare(
            geo_unet, normal_ctl, rgb_ctl, optimizer,
            train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_irs,
            lr_scheduler
        )
    train_dataloader = MixedDataLoaderN(
        [train_dataloader_hypersim,
        train_dataloader_irs],
        [7,5],
        seed=int(time.time())
    )



    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        unet.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        unet.to(dtype=weight_dtype)

    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def _tb_safe(v):
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            return v
        if v is None:
            return "None"
        return str(v)

    if accelerator.is_main_process:
        if hasattr(args, "items"):  # OmegaConf / DictConfig
            tracker_config = {k: _tb_safe(v) for k, v in args.items()}
        else:  # argparse Namespace
            tracker_config = {k: _tb_safe(v) for k, v in vars(args).items()}
    tb_writer = None
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

        if args.report_to == "tensorboard":
            for tracker in accelerator.trackers:
                if getattr(tracker, "name", "") == "tensorboard":
                    tb_writer = tracker.writer
                    break     
                 

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataloader.set_epoch(epoch)
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            pred_normal_m11_vis = None
            estimation_loss_ang_norm = torch.zeros((), device=device, dtype=torch.float32)
            disp_vis = None
            with accelerator.accumulate(geo_unet, normal_ctl, rgb_ctl):
                left_m11 = ensure_bchw(batch["left_rgb"]).to(
                    device=accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                right_m11 = ensure_bchw(batch["right_rgb"]).to(
                    device=accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                normal_gt_m11 = ensure_bchw(batch["normals"]).to(
                    device=accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                mask_norm_bool = batch["val_mask"].bool().to(device=accelerator.device)

                normal_gt_m11 = normalize_normals(normal_gt_m11).clamp(-1.0, 1.0)
                normal_gt_m11 [:, 0:1, :, :] = -normal_gt_m11[:, 0:1, :, :]
                normal_gt_m11 = normalize_normals(normal_gt_m11).clamp(-1.0, 1.0)
                mask_norm_bool = mask_to_bool(mask_norm_bool, size_hw=normal_gt_m11.shape[-2:])

                left_01 = m11_to_01(left_m11)
                right_01 = m11_to_01(right_m11)
                left_stereo = normalize_image_imagenet(left_01)
                right_stereo = normalize_image_imagenet(right_01)

                bsz = left_m11.shape[0]
                prompt_embeds_fp32 = prompt_embeds_1.repeat(bsz, 1, 1).to(device=device, dtype=torch.float32)
                prompt_embeds = prompt_embeds_fp32.to(dtype=mp_dtype)

                with torch.no_grad(), torch.autocast(device_type=amp_device_type, enabled=False):
                    _, disp_preds, _ = foundation(
                        left_stereo.float(),
                        right_stereo.float(),
                        iters=15, test_mode=False
                    )

                    disp_pred = disp_preds[-1].float().clamp(min=1e-3)
                    disp_pred_safe = disp_pred.clamp(min=1e-3, max=float(foundation_args.max_disp))
                    disp_vis = scalar01(disp_pred_safe[:1])

                    normal_init = disp_to_normal_points_no_intrinsics(
                        disp_pred_safe.detach(),
                        max_disp=float(foundation_args.max_disp),
                        use_log=True,
                        smooth_ks=5,
                        focal_scale=1.5,
                        flip_x=False,
                        flip_y=False,
                        flip_z=False,
                    )
                    valid_normal = (disp_pred_safe > 1e-3).float()
                    normal_init = normalize_normals(normal_init) * valid_normal

                right_warpm11 = warp_right_to_left(
                    right_m11,
                    disp_pred_safe.detach(),
                    sign=args.warp_sign,
                    align_corners=True,
                ).to(right_m11.dtype)
                right_warpm11_vis = safe_vis(right_warpm11[:1])

                # -------------------------------------------------------------
                # GeoStereo diffusion input:
                #   UNet: [RGB latent, noisy GT-normal latent]
                #
                # Warp ControlNet:
                #   warp-to-left latent only
                # Foundation-normal ControlNet:
                #   FoundationStereo normal-init latent only
                # -------------------------------------------------------------
                with torch.no_grad():
                    rgb_latent = encode_vae_latent(
                        vae,
                        left_m11,
                        sampling=args.vae_latent_sampling,
                    )
                    target_latent = encode_vae_latent(
                        vae,
                        normal_gt_m11.to(dtype=weight_dtype),
                        sampling=args.vae_latent_sampling,
                    )
                    init_latent = encode_vae_latent(
                        vae,
                        normal_init.to(dtype=weight_dtype),
                        sampling=args.vae_latent_sampling,
                    )
                    warp_to_left_latent = encode_vae_latent(
                        vae,
                        right_warpm11,
                        sampling=args.vae_latent_sampling,
                    )

                rgb_latent = rgb_latent.to(device=device, dtype=mp_dtype)
                target_latent = target_latent.to(device=device, dtype=mp_dtype)
                init_latent = init_latent.to(device=device, dtype=mp_dtype)
                warp_to_left_latent = warp_to_left_latent.to(device=device, dtype=mp_dtype)

                timesteps = torch.full(
                    (bsz,),
                    args.timestep,
                    device=device,
                    dtype=torch.long,
                )

                noise = torch.randn_like(target_latent)
                if args.noise_offset:
                    noise = noise + args.noise_offset * torch.randn(
                        (
                            target_latent.shape[0],
                            target_latent.shape[1],
                            1,
                            1,
                        ),
                        device=target_latent.device,
                        dtype=target_latent.dtype,
                    )

                noise_for_input = noise
                if args.input_perturbation:
                    noise_for_input = noise + args.input_perturbation * torch.randn_like(noise)

                noisy_target_latent = noise_scheduler.add_noise(
                    target_latent,
                    noise_for_input,
                    timesteps,
                )

                unet_input = torch.cat(
                    [rgb_latent, noisy_target_latent],
                    dim=1,
                )
                # The warp branch receives only the right image warped to the
                # left view.  The FoundationStereo normal is handled by the
                # separate normal ControlNet below.
                warp_controlnet_input = warp_to_left_latent.detach()
                normal_controlnet_input = init_latent.detach()

                if unet_input.shape[1] != 8:
                    raise RuntimeError(
                        f"Expected 8-channel UNet input, got {tuple(unet_input.shape)}"
                    )
                if warp_controlnet_input.shape[1] != 4:
                    raise RuntimeError(
                        "Expected a 4-channel warp ControlNet input, got "
                        f"{tuple(warp_controlnet_input.shape)}"
                    )
                if normal_controlnet_input.shape[1] != 4:
                    raise RuntimeError(
                        "Expected a 4-channel normal ControlNet input, got "
                        f"{tuple(normal_controlnet_input.shape)}"
                    )

                with accelerator.autocast():
                    # The geometry prior is static; timestep zero matches the
                    # original semantic-control behavior and prevents the
                    # condition itself from being interpreted as x_t.
                    normal_timesteps = torch.zeros_like(timesteps)
                    normal_down_res, _ = normal_ctl(
                        sample=normal_controlnet_input,
                        timestep=normal_timesteps,
                        encoder_hidden_states=prompt_embeds,
                        conditioning_scale=args.normal_control_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                    normal_down_res_copy = [
                        residual.to(dtype=mp_dtype) for residual in normal_down_res
                    ]

                    down_res, mid_res = rgb_ctl(
                        sample=warp_controlnet_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        conditioning_scale=args.warp_control_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                    down_res = [r.to(dtype=mp_dtype) for r in down_res]
                    mid_res = mid_res.to(dtype=mp_dtype)
                    task_emb = torch.tensor([1, 0], device=device, dtype=torch.float32).unsqueeze(0)
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
                    task_emb = task_emb.repeat(bsz, 1).to(dtype=mp_dtype)
                    model_pred = geo_unet(
                        sample=unet_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_res,
                        mid_block_additional_residual=mid_res,
                        dino_down_block_additional_residuals=(
                            normal_down_res_copy
                        ),
                        class_labels=task_emb,
                    )

                    loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    latent_loss = torch.zeros((), device=device, dtype=torch.float32)
                    estimation_loss_ang_norm = torch.zeros((), device=device, dtype=torch.float32)
                    if mask_norm_bool.any():
                        # `model_pred` is x0: the clean target-normal latent.
                        # `target_latent` was encoded once above and must NOT be
                        # multiplied by vae.config.scaling_factor a second time.
                        # 1) latent x0 regression loss
                        latent_loss = F.mse_loss(model_pred.float(), target_latent.float(), reduction="none")
                        latent_loss = latent_loss.mean(dim=1, keepdim=True)  # [B,1,h,w]

                        latent_mask = mask_to_bool(mask_norm_bool, size_hw=model_pred.shape[-2:]).float()
                        latent_loss = (latent_loss * latent_mask).sum() / (latent_mask.sum() + 1e-6)

                        # 2) Decode for angular supervision / visualization
                        loss = latent_loss

                        normal_cosine_loss = torch.zeros(
                            (),
                            device=device,
                            dtype=torch.float32,
                        )
                        loss = latent_loss

                        if args.lambda_ang > 0:
                            # 禁止外层 BF16 autocast 影响 VAE decoder
                            with torch.amp.autocast(
                                device_type="cuda",
                                enabled=False,
                            ):
                                # 不能 detach，否则 cosine loss 无法监督 UNet
                                pred_normal_m11 = vae.decode(
                                    model_pred.float()
                                    / float(vae.config.scaling_factor),
                                    return_dict=False,
                                )[0]

                                pred_normal_m11_vis = pred_normal_m11.detach()

                                normal_cosine_loss = stable_cosine_normal_loss(
                                    pred_normal=pred_normal_m11,
                                    gt_normal=normal_gt_m11.float(),
                                    valid_mask=mask_norm_bool,
                                )

                            # 前500步不加，随后1500步逐渐提高权重
                            warmup_ratio = min(
                                1.0,
                                max(0.0, float(global_step - 500)) / 1500.0,
                            )

                            current_normal_weight = (
                                args.lambda_ang * warmup_ratio
                            )

                            loss = (
                                latent_loss
                                + current_normal_weight * normal_cosine_loss
                            )

                        # 兼容你原来的日志变量
                        estimation_loss_ang_norm = normal_cosine_loss

                #avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean()

                #avg_loss = accelerator.gather(loss.detach().repeat(bsz)).mean()


                #train_loss += avg_loss.item() / args.gradient_accumulation_steps

                train_loss += loss.detach().float().item() / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    avg_loss = accelerator.gather(loss.detach().reshape(1)).mean()
                else:
                    avg_loss = loss.detach()





                accelerator.backward(loss)

                # ---------------------------------------------------------
                # Non-finite-gradient protection.
                # If any rank encounters NaN/Inf, all ranks discard this
                # micro-step so DDP model/optimizer states remain consistent.
                # ---------------------------------------------------------
                bad_grad_name = None

                named_trainable_params = (
                    [
                        (f"geo_unet.{name}", param)
                        for name, param in geo_unet.named_parameters()
                    ]
                    + [
                        (f"normal_ctl.{name}", param)
                        for name, param in normal_ctl.named_parameters()
                    ]
                    + [
                        (f"rgb_ctl.{name}", param)
                        for name, param in rgb_ctl.named_parameters()
                    ]
                )

                for name, param in named_trainable_params:
                    if (
                        param.grad is not None
                        and not torch.isfinite(param.grad).all()
                    ):
                        bad_grad_name = name
                        break

                local_bad_grad = torch.tensor(
                    int(bad_grad_name is not None),
                    device=accelerator.device,
                    dtype=torch.int32,
                )
                global_bad_grad = accelerator.reduce(
                    local_bad_grad,
                    reduction="max",
                )

                if global_bad_grad.item() > 0:
                    accelerator.print(
                        "[SKIP STEP] non-finite gradient detected; "
                        f"local parameter={bad_grad_name}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if accelerator.sync_gradients:
                    params_to_clip = (
                        list(geo_unet.parameters())
                        + list(normal_ctl.parameters())
                        + list(rgb_ctl.parameters())
                    )

                    grad_norm = accelerator.clip_grad_norm_(
                        params_to_clip,
                        args.max_grad_norm,
                    )

                    local_bad_norm = torch.tensor(
                        int(not torch.isfinite(grad_norm).all().item()),
                        device=accelerator.device,
                        dtype=torch.int32,
                    )
                    global_bad_norm = accelerator.reduce(
                        local_bad_norm,
                        reduction="max",
                    )

                    if global_bad_norm.item() > 0:
                        accelerator.print(
                            "[SKIP STEP] non-finite global gradient norm: "
                            f"{grad_norm.detach().float().item()}"
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                log_dict = {
                    "train/loss_total": loss.detach().float().item(),
                    "train/loss_latent": latent_loss.detach().float().item(),
                    "train/loss_ang": estimation_loss_ang_norm.detach().float().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(log_dict, step=global_step)

                if tb_writer is not None and (global_step % args.vis_step == 0):
                    tb_writer.add_image("diff/left_rgb", to_grid01(safe_vis(m11_to_01(left_m11[:1]))), global_step)
                    tb_writer.add_image("diff/right_rgb", to_grid01(safe_vis(m11_to_01(right_m11[:1]))), global_step)
                    tb_writer.add_image("diff/right_warp_to_left", to_grid01(safe_vis(m11_to_01(right_warpm11_vis))), global_step)

                    tb_writer.add_image("diff/normal_init", to_grid01(safe_vis(norm_to_01(normal_init[:1]))), global_step)
                    tb_writer.add_image("diff/normal_gt", to_grid01(safe_vis(norm_to_01(normal_gt_m11[:1]))), global_step)
                    tb_writer.add_image("diff/valid_mask", to_grid01(safe_vis(mask_norm_bool[:1].float())), global_step)

                    noisy_target_vis = vae.decode(
                        noisy_target_latent[:1] / vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    tb_writer.add_image(
                        "diff/noisy_target_normal_decoded",
                        to_grid01(safe_vis(norm_to_01(noisy_target_vis))),
                        global_step,
                    )

                    if disp_vis is not None:
                        tb_writer.add_image("diff/disp_pred", to_grid01(safe_vis(disp_vis)), global_step)

                    if pred_normal_m11_vis is not None:
                        pred_normal_m11_vis = normalize_normals(pred_normal_m11_vis.float()).to(pred_normal_m11_vis.dtype)
                        tb_writer.add_image(
                            "diff/pred_normal_decoded",
                            to_grid01(safe_vis(norm_to_01(pred_normal_m11_vis[:1]))),
                            global_step,
                        )

                    tb_writer.flush()

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    logger.info(
                        f"Entered Saving Code at global step {global_step} checkpointing_steps {args.checkpointing_steps}"
                    )
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        os.makedirs(save_path, exist_ok=True)

                        geo_unet_to_save = accelerator.unwrap_model(
                            geo_unet
                        )
                        unet_to_save = geo_unet_to_save.unet
                        normal_ctl_to_save = unwrap_model(normal_ctl)
                        rgb_ctl_to_save = unwrap_model(rgb_ctl)

                        unet_to_save.save_pretrained(
                            os.path.join(
                                save_path,
                                "unet",
                            )
                        )
                        normal_ctl_to_save.save_pretrained(os.path.join(save_path, "normal_controlnet"))
                        rgb_ctl_to_save.save_pretrained(os.path.join(save_path, "controlnet"))

                        vae.save_pretrained(os.path.join(save_path, "vae"))
                        text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
                        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
                        noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))

                        logger.info(f"Saved GeoStereo training components to {save_path}; ""FoundationStereo is not duplicated.")
                        logger.info(f"Saved model components to {save_path}")

                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    logger.info(f"Saved full Accelerate state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        geo_unet_to_save = unwrap_model(geo_unet)
        unet_to_save = geo_unet_to_save.unet
        normal_ctl_to_save = unwrap_model(normal_ctl)
        rgb_ctl_to_save = unwrap_model(rgb_ctl)

        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Saving GeoStereo components to {args.output_dir}; ""FoundationStereo is not duplicated.")

        unet_to_save.save_pretrained(os.path.join(args.output_dir, "unet"))
        normal_ctl_to_save.save_pretrained(os.path.join(args.output_dir, "normal_controlnet"))
        rgb_ctl_to_save.save_pretrained(os.path.join(args.output_dir, "controlnet"))

        vae.save_pretrained(os.path.join(args.output_dir, "vae"))
        text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))


    logger.info("Finished training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
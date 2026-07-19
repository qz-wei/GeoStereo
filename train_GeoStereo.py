import argparse
import logging
import math
import os
import shutil
import sys
import accelerate
import datasets
import torch
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
from GeoStereo.depthanything_head import DepthAnythingFeature
from util.utils import  disp_to_normal_points_no_intrinsics, set_requires_grad, unwrap,dino_unet_forward,DummyStereoDataset,warp_right_to_left
import torchvision
from dataloaders.load_stereo import HypersimStereoJSONL,IRSStereo,VirtualKITTI2Stereo,MixedDataLoaderN
from util.loss import AngularLoss
from util.lr_scheduler import IterExponential
from util.unet_prep import replace_unet_conv_in
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import AutoencoderKL, UNet2DConditionModel,ControlNetModel
from GeoStereo.controlnetvae import ControlNetVAEModel
from GeoStereo.dino_controlnetvae import DINOControlNetVAEModel
from GeoStereo.core.foundation_stereo import FoundationStereo
import time

if is_wandb_available():
    import wandb


check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def perturb_normal_init(
    normal_init,
    valid_mask,
    p_clean=0.30,
    p_gauss=0.40,
    p_blur=0.15,
    p_drop=0.15,
    sigma_range=(0.02, 0.08),
    drop_ratio_range=(0.05, 0.25),
    blur_ks=5,
):
    """
    normal_init: [B, 3, H, W], range in [-1, 1]
    valid_mask:  [B, 1, H, W] or [B, H, W]
    """
    x = normal_init.clone()

    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.float()

    r = torch.rand(1, device=x.device).item()

    if r < p_clean:
        # 保持原样
        out = x

    elif r < p_clean + p_gauss:
        # 小高斯扰动，再归一化
        sigma = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(*sigma_range)
        noise = torch.randn_like(x) * sigma
        out = x + noise * valid_mask
        out = F.normalize(out, dim=1, eps=1e-6)

    elif r < p_clean + p_gauss + p_blur:
        # 模拟“过平滑”的 dirty init
        out = F.avg_pool2d(x, kernel_size=blur_ks, stride=1, padding=blur_ks // 2)
        out = F.normalize(out, dim=1, eps=1e-6)

    else:
        # 随机丢一些区域，逼模型不要过度依赖 init
        drop_ratio = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(*drop_ratio_range)
        keep = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > drop_ratio).float()
        out = x * keep
        out = F.normalize(out, dim=1, eps=1e-6)

    out = out * valid_mask
    return out.clamp(-1.0, 1.0)

def expand_lotus_unet_conv_in_4_to_8_tiny_extra(unet, extra_scale=1e-6):
    """
    将 Lotus 原始 4ch conv_in 扩成 8ch。

    前 4 通道：完全复制 Lotus pretrained conv_in。
    后 4 通道：用 Lotus conv_in 的权重乘以 1e-6 初始化。

    这样初始时：
        conv8([rgb_latent, init_latent])
        ≈ conv4(rgb_latent)

    但后 4 通道权重是 trainable 的，后续可以学习使用 normal_init。
    """
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

        # RGB latent 部分：完全继承 Lotus
        new_conv.weight[:, :4, :, :].copy_(old_conv.weight)

        # normal_init latent 部分：极小初始化，几乎不影响初始输出，但可以训练
        new_conv.weight[:, 4:, :, :].copy_(old_conv.weight * extra_scale)

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    unet.conv_in = new_conv
    unet.register_to_config(in_channels=8)
    return unet


def parse_args():
    parser = argparse.ArgumentParser(
        description="StableNormal training script: fixed training of UNet + DINO ControlNet + RGB ControlNet."
    )

    # Core paths / run config
    parser.add_argument("--model_path", type=str, default='/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/lotus')
    parser.add_argument("--output_dir", type=str, default="outputs/ckpt")
    parser.add_argument("--cfg_yaml", type=str, default='/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/cfg.yaml')
    parser.add_argument("--ckpt_dir", type=str, default='/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/foundation_stereo/model_best_bp2-001.pth')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int,  default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Optimizer / LR: keep first-script style
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=20000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Dataloader / logging / distributed
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_dir", type=str, default="outputs/logs")
    parser.add_argument("--mixed_precision", type=str, default='no', choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention",default=True, action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default="e2e-ft-diffusion")
    parser.add_argument("--mono", type=bool, default=False)
    parser.add_argument("--non_ema_revision", type=str, default=None)   
    parser.add_argument("--warp_sign", type=str, default="minus", choices=["minus", "plus"],
                   help="warp sign: minus means x_r = x_l - disp (common).")
    # Training-time controls used inside the current epoch logic
    #parser.add_argument("--train_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=416)
    parser.add_argument("--control_scale", type=float, default=0.2)
    parser.add_argument("--lambda_ang", type=float, default=0.2)
    parser.add_argument("--vis_step", type=int, default=50)
    

    args = parser.parse_args()
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
    # Load StableNormal pipeline and retrieve modules used by the current epoch
    # ---------------------------------------------------------------------
    restore_ckpt = '/file_system/goosefsx/whu/users/hong.li/qizhe/diffusion-e2e-ft-main/training/root1/ckpt/checkpoint-1400'
    tokenizer = CLIPTokenizer.from_pretrained(restore_ckpt, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(restore_ckpt, subfolder="text_encoder")
    text_encoder.to(device)
    vae = AutoencoderKL.from_pretrained(restore_ckpt, subfolder="vae")


    unet = UNet2DConditionModel.from_pretrained(restore_ckpt, subfolder="unet", revision=args.non_ema_revision,
        class_embed_type="projection", projection_class_embeddings_input_dim=4, # Important! new projection layer will be initailized.
        low_cpu_mem_usage=False, device_map=None,)
    rgb_ctl = ControlNetVAEModel.from_pretrained(restore_ckpt, subfolder="controlnet")
    dino_ctl = DINOControlNetVAEModel.from_pretrained(restore_ckpt, subfolder="dino_controlnet")

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(args.cfg_yaml)
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    cfg["mixed_precision"] = False     
    args1 = OmegaConf.create(cfg)
    logger.info(f"FoundationStereo Args (args1):\n{args1}")
    logger.info(f"Using pretrained model from {ckpt_dir}")

    depth_any = DepthAnythingFeature(
            encoder="vitl",
            tokens_out_dim=None,
            resize_to=None,
        )


    foundation = FoundationStereo(args1,depth_any)
    ckpt = torch.load(ckpt_dir)
    missing_keys, unexpected_keys = foundation.load_state_dict(ckpt['model'], strict=False)
    foundation = foundation.to(device=device, dtype=torch.float32)

    if len(missing_keys) > 0:
        print(f"⚠️ [警告] 缺少这些层的权重 (Missing Keys):\n{missing_keys}\n")
        
    if len(unexpected_keys) > 0:
        print(f"ℹ️ [提示] 检查点中多余的权重 (Unexpected Keys):\n{unexpected_keys}\n")

    foundation.eval()
    foundation.feature = foundation.feature.to(device=device, dtype=torch.float32)
    foundation.feature.dino = foundation.feature.dino.to(device=device, dtype=torch.float32)


    if rgb_ctl is None:
        raise ValueError("StableNormalPipeline.controlnet is None, but this script is configured to train rgb_ctl.")

    # Foundation is frozen and used in fp32
    #foundation.to(device=device, dtype=torch.float32)
    try:
        foundation.feature.dino.to(device=device, dtype=torch.float32)
    except Exception:
        pass
    try:
        if hasattr(foundation, "args") and hasattr(foundation.args1, "mixed_precision"):
            foundation.args1.mixed_precision = False
    except Exception:
        pass

    if unet.config["in_channels"] != 8:
        expand_lotus_unet_conv_in_4_to_8_tiny_extra(unet, extra_scale=1e-6)
        logger.info("Unet conv_in layer is replaced for 8-channel RGB+init-normal input")

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

    dino_ctl.train(True)
    set_requires_grad(dino_ctl, True)

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

    # Save/load state hooks for all trainable modules
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    m = unwrap_model(model)
                    if m is unwrap_model(unet):
                        m.save_pretrained(os.path.join(output_dir, "unet"))
                    elif m is unwrap_model(dino_ctl):
                        m.save_pretrained(os.path.join(output_dir, "dino_controlnet"))
                    elif m is unwrap_model(rgb_ctl):
                        m.save_pretrained(os.path.join(output_dir, "rgb_controlnet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                m = unwrap_model(model)

                if m is unwrap_model(unet):
                    load_model = type(m).from_pretrained(input_dir, subfolder="unet")
                elif m is unwrap_model(dino_ctl):
                    load_model = type(m).from_pretrained(input_dir, subfolder="dino_controlnet")
                elif m is unwrap_model(rgb_ctl):
                    load_model = type(m).from_pretrained(input_dir, subfolder="rgb_controlnet")
                else:
                    continue

                m.register_to_config(**load_model.config)
                m.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = torch.optim.AdamW(
        [
            {"params": unet.parameters()},
            {"params": dino_ctl.parameters()},
            {"params": rgb_ctl.parameters()},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_func = IterExponential(
        total_iter_length=args.lr_total_iter_length ,
        final_ratio=0.01,
        warmup_steps=args.lr_exp_warmup_steps,
    )
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)


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

    unet, dino_ctl, rgb_ctl, optimizer, \
        train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_irs, \
        lr_scheduler = accelerator.prepare(
            unet, dino_ctl, rgb_ctl, optimizer,
            train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_irs,
            lr_scheduler
        )
    train_dataloader = MixedDataLoaderN(
        [train_dataloader_hypersim,
        train_dataloader_vkitti,
        train_dataloader_irs],
        [7,1,5],
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

    vae.to(accelerator.device, dtype=weight_dtype)
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

    angular_loss_norm = AngularLoss()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataloader.set_epoch(epoch)
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            pred_normal_m11_vis = None
            estimation_loss_ang_norm = torch.zeros((), device=device, dtype=torch.float32)
            disp_vis = None
            with accelerator.accumulate(unet):
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
                    _, disp_preds,left_tokens = foundation(
                        left_stereo.float(),
                        right_stereo.float(),
                        iters=15, test_mode=False
                    )

                    disp_pred = disp_preds[-1].float().clamp(min=1e-3)
                    disp_pred_safe = disp_pred.clamp(min=1e-3, max=float(args1.max_disp))
                    disp_vis = scalar01(disp_pred_safe[:1])

                    normal_init = disp_to_normal_points_no_intrinsics(
                        disp_pred_safe.detach(),
                        max_disp=float(args1.max_disp),
                        use_log=True,
                        smooth_ks=5,
                        focal_scale=1.5,
                        flip_x=False,
                        flip_y=False,
                        flip_z=False,
                    )
                    valid_normal = (disp_pred_safe > 1e-3).float()
                    normal_init = normalize_normals(normal_init) * valid_normal
                    dino_tokens_raw = left_tokens.float()

                with torch.no_grad():
                    rgb_latent = vae.encode(left_m11).latent_dist.sample() * vae.config.scaling_factor
                    init_latent = (
                        vae.encode(normal_init.to(dtype=weight_dtype)).latent_dist.sample()
                        * vae.config.scaling_factor
                    )

                rgb_latent = rgb_latent.to(dtype=mp_dtype)
                init_latent = init_latent.to(dtype=mp_dtype)
                dino_tokens_raw = dino_tokens_raw.to(device=device, dtype=mp_dtype)

                right_warpm11 = warp_right_to_left(
                    right_m11, disp_pred_safe.detach(),
                    sign=args.warp_sign, align_corners=True
                ).to(right_m11.dtype)
                right_warpm11_vis = safe_vis(right_warpm11[:1])
                warp_to_left_latent = vae.encode(right_warpm11).latent_dist.sample() * vae.config.scaling_factor

                unet_input = torch.cat([rgb_latent, init_latent], dim=1)
                t_fixed = 999
                timesteps = torch.full((bsz,), t_fixed, device=device, dtype=torch.long)
                dino_ctl_model = unwrap_model(dino_ctl)
                with torch.autocast(device_type=amp_device_type, enabled=False):
                    dino_feat = dino_ctl_model.dino_controlnet_cond_embedding(dino_tokens_raw.float())
                    if dino_feat.shape[-2:] != unet_input.shape[-2:]:
                        dino_feat = F.interpolate(
                            dino_feat,
                            size=unet_input.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    dino_down_res, _ = dino_ctl(
                        dino_feat,
                        timesteps,
                        encoder_hidden_states=prompt_embeds_fp32,
                        conditioning_scale=args.control_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                dino_down_res_copy = [r.to(dtype=mp_dtype) for r in dino_down_res]

                with accelerator.autocast():
                    down_res, mid_res = rgb_ctl(
                        warp_to_left_latent.detach(),
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        conditioning_scale=args.control_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                    down_res = [r.to(dtype=mp_dtype) for r in down_res]
                    mid_res = mid_res.to(dtype=mp_dtype)
                    task_emb = torch.tensor([1, 0], device=device, dtype=torch.float32).unsqueeze(0)
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
                    task_emb = task_emb.repeat(bsz, 1).to(dtype=mp_dtype)
                    model_pred = dino_unet_forward(
                        unwrap(unet),
                        unet_input,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_res,
                        mid_block_additional_residual=mid_res,
                        dino_down_block_additional_residuals=dino_down_res_copy,
                        class_labels=task_emb,
                        return_dict=False,
                    )[0]

                    loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    latent_loss = torch.zeros((), device=device, dtype=torch.float32)
                    estimation_loss_ang_norm = torch.zeros((), device=device, dtype=torch.float32)
                    if mask_norm_bool.any():
                        with torch.no_grad():
                            target_latent = vae.encode(normal_gt_m11.to(dtype=weight_dtype)).latent_dist.sample()
                            target_latent = target_latent * vae.config.scaling_factor

                        # 1) Lotus-style latent regression loss
                        latent_loss = F.mse_loss(model_pred.float(), target_latent.float(), reduction="none")
                        latent_loss = latent_loss.mean(dim=1, keepdim=True)  # [B,1,h,w]

                        latent_mask = mask_to_bool(mask_norm_bool, size_hw=model_pred.shape[-2:]).float()
                        latent_loss = (latent_loss * latent_mask).sum() / (latent_mask.sum() + 1e-6)

                        # 2) Decode for angular supervision / visualization
                        pred_normal_m11 = vae.decode(
                            model_pred / vae.config.scaling_factor,
                            return_dict=False,
                        )[0]
                        pred_normal_m11_vis = pred_normal_m11.detach()

                        pred_normal_m11 = pred_normal_m11.float()
                        pred_normal_m11 = pred_normal_m11 / (torch.norm(pred_normal_m11, p=2, dim=1, keepdim=True) + 1e-5)
                        pred_normal_m11 = torch.clamp(pred_normal_m11, -1.0, 1.0)

                        estimation_loss_ang_norm = angular_loss_norm(
                            pred_normal_m11, normal_gt_m11.float(), mask_norm_bool
                        )

                        if torch.isnan(estimation_loss_ang_norm).any() or torch.isinf(estimation_loss_ang_norm).any():
                            logger.warning(f"NaN/Inf angular loss at global_step={global_step}")
                            loss = latent_loss
                        else:
                            loss = latent_loss  + args.lambda_ang* estimation_loss_ang_norm 

                #avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean()

                #avg_loss = accelerator.gather(loss.detach().repeat(bsz)).mean()


                #train_loss += avg_loss.item() / args.gradient_accumulation_steps

                train_loss += loss.detach().float().item() / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    avg_loss = accelerator.gather(loss.detach().reshape(1)).mean()
                else:
                    avg_loss = loss.detach()





                accelerator.backward(loss)


                # 检查梯度
                bad_grad = None
                for name, p in list(unet.named_parameters()) + list(dino_ctl.named_parameters()) + list(rgb_ctl.named_parameters()):
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            bad_grad = name
                            print(
                                "[BAD GRAD]",
                                name,
                                torch.isnan(p.grad).any().item(),
                                torch.isinf(p.grad).any().item(),
                                p.grad.detach().float().amin().item(),
                                p.grad.detach().float().amax().item(),
                                flush=True,
                            )
                            break


                if accelerator.sync_gradients:
                    params_to_clip = (
                        list(unet.parameters()) + list(dino_ctl.parameters()) + list(rgb_ctl.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


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

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        unet_to_save = unwrap_model(unet)
                        dino_ctl_to_save = unwrap_model(dino_ctl)
                        rgb_ctl_to_save = unwrap_model(rgb_ctl)

                        unet_to_save.save_pretrained(os.path.join(save_path, "unet"))
                        dino_ctl_to_save.save_pretrained(os.path.join(save_path, "dino_controlnet"))
                        rgb_ctl_to_save.save_pretrained(os.path.join(save_path, "controlnet"))

                        vae.save_pretrained(os.path.join(save_path, "vae"))
                        text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
                        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

                        logger.info(f"Saved model-only checkpoint to {save_path}")
                        #accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_to_save = unwrap_model(unet)
        dino_ctl_to_save = unwrap_model(dino_ctl)
        rgb_ctl_to_save = unwrap_model(rgb_ctl)

        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Saving modules to {args.output_dir}")

        unet_to_save.save_pretrained(os.path.join(args.output_dir, "unet"))
        dino_ctl_to_save.save_pretrained(os.path.join(args.output_dir, "dino_controlnet"))
        rgb_ctl_to_save.save_pretrained(os.path.join(args.output_dir, "controlnet"))

        #vae.save_pretrained(os.path.join(args.output_dir, "vae"))
        #text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
        #tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))


    logger.info("Finished training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import is_torch_version
from torch.utils.data import Dataset
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers


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


def unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag



class DummyStereoDataset(Dataset):
    def __init__(self, length=64, image_size=(224, 224), mono=False):
        self.length = int(length)
        self.H, self.W = int(image_size[0]), int(image_size[1])
        self.mono = mono

        # 建议用 56 的倍数；448 同时兼容 DINO(14) 和 VAE(8)
        assert self.H % 56 == 0 and self.W % 56 == 0, \
            f"image_size should be multiple of 56, got {(self.H, self.W)}"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # left/right: 假装已经是 dataloader 输出后的 [-1, 1]
        left = torch.rand(3, self.H, self.W, dtype=torch.float32) * 2.0 - 1.0

        if self.mono:
            right = left.clone()
        else:
            right = (left + 0.02 * torch.randn_like(left)).clamp(-1.0, 1.0)

        # normals: 先随机，再单位化到 [-1,1] 语义
        normals = torch.randn(3, self.H, self.W, dtype=torch.float32)
        normals = normals / (torch.norm(normals, p=2, dim=0, keepdim=True) + 1e-6)
        normals = normals.clamp(-1.0, 1.0)

        # val_mask: 全有效
        val_mask = torch.ones(self.H, self.W, dtype=torch.bool)

        return {
            "left": left,
            "right": right,
            "normals": normals,
            "val_mask": val_mask,
        }
    
    

def dino_unet_forward(
    unet,
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
    dino_down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    default_overall_up_factor = 2 ** unet.num_upsamplers

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

    if unet.config.center_input_sample:
        sample = 2 * sample - 1.0

    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    class_emb = unet.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if unet.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = unet.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if unet.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    encoder_hidden_states = unet.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    sample = unet.conv_in(sample)

    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": unet.position_net(**gligen_args)}

    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(unet, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    is_adapter = down_intrablock_additional_residuals is not None

    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    def residual_downforward(
        block,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        additional_residuals: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        for resnet in block.resnets:
            if block.training and block.gradient_checkpointing:

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
                hidden_states = hidden_states + additional_residuals.pop(0)

            output_states = output_states + (hidden_states,)

        if block.downsamplers is not None:
            for downsampler in block.downsamplers:
                hidden_states = downsampler(hidden_states)
                hidden_states = hidden_states + additional_residuals.pop(0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    def residual_blockforward(
        block,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        additional_residuals: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        blocks = list(zip(block.resnets, block.attentions))

        for resnet, attn in blocks:
            if block.training and block.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
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

            hidden_states = hidden_states + additional_residuals.pop(0)
            output_states = output_states + (hidden_states,)

        if block.downsamplers is not None:
            for downsampler in block.downsamplers:
                hidden_states = downsampler(hidden_states)
                hidden_states = hidden_states + additional_residuals.pop(0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    if dino_down_block_additional_residuals is None:
        raise ValueError("dino_down_block_additional_residuals is None")

    down_intrablock_additional_residuals = list(dino_down_block_additional_residuals)

    sample = sample + down_intrablock_additional_residuals.pop(0)
    down_block_res_samples = (sample,)

    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = residual_blockforward(
                downsample_block,
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                additional_residuals=down_intrablock_additional_residuals,
            )
        else:
            sample, res_samples = residual_downforward(
                downsample_block,
                hidden_states=sample,
                temb=emb,
                additional_residuals=down_intrablock_additional_residuals,
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

    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = unet.mid_block(sample, emb)

        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample = sample + down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    for i, upsample_block in enumerate(unet.up_blocks):
        is_final_block = i == len(unet.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

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

    if unet.conv_norm_out:
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(unet, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)


def warp_right_to_left(
    right_bchw: torch.Tensor,
    disp_b1hw: torch.Tensor,
    sign: str = "minus",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Warp right image to left view using left disparity.
    right_bchw: [B,C,H,W]
    disp_b1hw:  [B,1,H,W], disparity in pixels at left frame resolution
    """
    orig_dtype = right_bchw.dtype

    # geometry ops in fp32 for stability / dtype consistency
    right_bchw_f = right_bchw.float()
    disp_b1hw_f = disp_b1hw.float()

    B, C, H, W = right_bchw_f.shape
    xs = torch.linspace(0, W - 1, W, device=right_bchw_f.device, dtype=torch.float32)
    ys = torch.linspace(0, H - 1, H, device=right_bchw_f.device, dtype=torch.float32)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # H,W
    X = X[None, None].expand(B, 1, H, W)
    Y = Y[None, None].expand(B, 1, H, W)

    if sign == "minus":
        Xr = X - disp_b1hw_f
    elif sign == "plus":
        Xr = X + disp_b1hw_f
    else:
        raise ValueError("sign must be 'minus' or 'plus'")

    grid_x = 2.0 * (Xr / (W - 1.0)) - 1.0
    grid_y = 2.0 * (Y  / (H - 1.0)) - 1.0
    grid = torch.cat([grid_x, grid_y], dim=1)      # [B,2,H,W]
    grid = grid.permute(0, 2, 3, 1).contiguous()   # [B,H,W,2]

    warped = F.grid_sample(
        right_bchw_f,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return warped.to(orig_dtype)
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

"""Geometry ControlNet used by GeoStereo normal estimation.

This is a drop-in replacement for the original ``dino_controlnetvae.py``.
The class name ``Normal_ControlNetVAEModel`` is intentionally retained so old
training scripts/checkpoints can still be loaded.  The conditioning branch can
be reconfigured from the original DINO feature input (typically 1024 channels,
with x2 upsampling) to a FoundationStereo normal VAE latent (4 channels, no
spatial upsampling).
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.models import ControlNetModel
from diffusers.models.controlnet import ControlNetOutput


def conv_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported convolution dimensionality: {dims}")


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class Normal_ControlNetConditioningEmbedding(nn.Module):
    """Project an external condition to the first ControlNet feature width.

    ``up_sampling='transpose'`` reproduces the original StableNormal DINO
    branch, whose DINO feature map is half the latent spatial resolution.

    ``up_sampling='none'`` is used for a FoundationStereo normal VAE latent,
    which already has the same spatial resolution as the diffusion latent.
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 64, 128),
        up_sampling: str = "transpose",
        zero_init_out: bool = True,
    ):
        super().__init__()

        if not block_out_channels:
            raise ValueError("block_out_channels must contain at least one value")
        if up_sampling not in {"transpose", "none", "identity"}:
            raise ValueError(
                "up_sampling must be one of {'transpose', 'none', 'identity'}, "
                f"got {up_sampling!r}"
            )

        self.conditioning_channels = int(conditioning_channels)
        self.conditioning_embedding_channels = int(conditioning_embedding_channels)
        self.up_sampling = up_sampling

        self.conv_in = nn.Conv2d(
            self.conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList()
        for index in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[index]
            channel_out = block_out_channels[index + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
            )

        if up_sampling == "transpose":
            conv_out: nn.Module = nn.ConvTranspose2d(
                block_out_channels[-1],
                self.conditioning_embedding_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            conv_out = nn.Conv2d(
                block_out_channels[-1],
                self.conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )

        self.conv_out = zero_module(conv_out) if zero_init_out else conv_out

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 4:
            raise ValueError(
                "Conditioning input must be BCHW, "
                f"got shape={tuple(conditioning.shape)}"
            )
        if conditioning.shape[1] != self.conditioning_channels:
            raise ValueError(
                f"Expected {self.conditioning_channels} conditioning channels, "
                f"got shape={tuple(conditioning.shape)}"
            )

        embedding = F.silu(self.conv_in(conditioning))
        for block in self.blocks:
            embedding = F.silu(block(embedding))
        return self.conv_out(embedding)


class Normal_ControlNetVAEModel(ControlNetModel):
    """ControlNet whose input is an externally supplied feature/latent map.

    Backward compatibility:
      * old DINO checkpoints can be loaded unchanged;
      * call :meth:`reconfigure_for_normal_latent` once after loading an old
        checkpoint to replace the DINO projector by a 4-channel normal-latent
        projector;
      * newly saved checkpoints preserve the new projector configuration.

    During ``forward`` the model accepts either:
      * a raw condition whose channel count equals
        ``dino_conditioning_channels``; it is projected internally; or
      * an already projected feature map with ``block_out_channels[0]``
        channels, matching the legacy call path.

    Keeping the projection inside ``forward`` is important for DDP: all
    trainable conditioning-embedding parameters are then exercised through
    the wrapped module's forward pass.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool, ...]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (
            16,
            32,
            96,
            256,
        ),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        dino_up_sampling: str = "transpose",
        dino_conditioning_embedding_channels: int = 320,
        dino_conditioning_channels: int = 1024,
        dino_block_out_channels: Tuple[int, ...] = (512, 128, 256, 256),
        dino_zero_init_out: bool = True,
    ):
        super().__init__(
            in_channels,
            conditioning_channels,
            flip_sin_to_cos,
            freq_shift,
            down_block_types,
            mid_block_type,
            only_cross_attention,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            transformer_layers_per_block,
            encoder_hid_dim,
            encoder_hid_dim_type,
            attention_head_dim,
            num_attention_heads,
            use_linear_projection,
            class_embed_type,
            addition_embed_type,
            addition_time_embed_dim,
            num_class_embeds,
            upcast_attention,
            resnet_time_scale_shift,
            projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels,
            global_pool_conditions,
            addition_embed_type_num_heads,
        )

        if dino_conditioning_embedding_channels != block_out_channels[0]:
            raise ValueError(
                "dino_conditioning_embedding_channels must match "
                f"block_out_channels[0] ({block_out_channels[0]}), got "
                f"{dino_conditioning_embedding_channels}"
            )

        self.dino_controlnet_cond_embedding = Normal_ControlNetConditioningEmbedding(
            up_sampling=dino_up_sampling,
            conditioning_embedding_channels=dino_conditioning_embedding_channels,
            conditioning_channels=dino_conditioning_channels,
            block_out_channels=tuple(dino_block_out_channels),
            zero_init_out=dino_zero_init_out,
        )

    def reconfigure_for_normal_latent(
        self,
        conditioning_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
        zero_init_out: bool = True,
    ) -> "Normal_ControlNetVAEModel":
        """Replace the original DINO projector with a normal-latent projector.

        Call this after loading an old StableNormal DINO checkpoint. The old
        ControlNet down-block weights are retained; only the input projector is
        replaced because its input semantics and channel count have changed.
        """

        reference_parameter = next(self.parameters())
        embedding_channels = int(self.config.block_out_channels[0])

        new_embedding = Normal_ControlNetConditioningEmbedding(
            conditioning_embedding_channels=embedding_channels,
            conditioning_channels=conditioning_channels,
            block_out_channels=tuple(block_out_channels),
            up_sampling="none",
            zero_init_out=zero_init_out,
        ).to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        self.dino_controlnet_cond_embedding = new_embedding
        self.register_to_config(
            dino_up_sampling="none",
            dino_conditioning_embedding_channels=embedding_channels,
            dino_conditioning_channels=int(conditioning_channels),
            dino_block_out_channels=list(block_out_channels),
            dino_zero_init_out=bool(zero_init_out),
        )
        return self

    def _prepare_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 4:
            raise ValueError(f"sample must be BCHW, got shape={tuple(sample.shape)}")

        projected_channels = int(self.config.block_out_channels[0])
        conditioning_channels = int(self.config.dino_conditioning_channels)

        if sample.shape[1] == conditioning_channels:
            sample = self.dino_controlnet_cond_embedding(sample)
        elif sample.shape[1] != projected_channels:
            raise ValueError(
                "Geometry ControlNet expected either a raw condition with "
                f"{conditioning_channels} channels or a projected feature with "
                f"{projected_channels} channels, got shape={tuple(sample.shape)}"
            )
        return sample

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], Optional[torch.Tensor]]]:
        # Prefer an explicitly supplied condition; otherwise ``sample`` itself
        # is treated as either a raw condition or a legacy pre-projected feature.
        if controlnet_cond is not None:
            sample = controlnet_cond
        sample = self._prepare_sample(sample)

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        else:
            timesteps = timesteps.to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided when class embedding is enabled")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            emb = emb + self.class_embedding(class_labels).to(dtype=emb.dtype)

        aug_emb = None
        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                if added_cond_kwargs is None:
                    raise ValueError("added_cond_kwargs is required for addition_embed_type='text_time'")
                if "text_embeds" not in added_cond_kwargs or "time_ids" not in added_cond_kwargs:
                    raise ValueError("text_embeds and time_ids are required for text_time conditioning")
                text_embeds = added_cond_kwargs["text_embeds"]
                time_ids = added_cond_kwargs["time_ids"]
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                aug_emb = self.add_embedding(
                    torch.cat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
                )
        if aug_emb is not None:
            emb = emb + aug_emb

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                )
            down_block_res_samples += res_samples

        # The first residual is already zero at initialization because the
        # conditioning projector's final convolution is zero-initialized.
        controlnet_down_block_res_samples = (down_block_res_samples[0],)
        for residual, controlnet_block in zip(
            down_block_res_samples[1:],
            self.controlnet_down_blocks[1:],
        ):
            controlnet_down_block_res_samples += (controlnet_block(residual),)

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = None

        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(
                -1,
                0,
                len(down_block_res_samples),
                device=sample.device,
                dtype=sample.dtype,
            ) * conditioning_scale
            down_block_res_samples = [
                residual * scale
                for residual, scale in zip(down_block_res_samples, scales)
            ]
        else:
            down_block_res_samples = [
                residual * conditioning_scale for residual in down_block_res_samples
            ]

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(residual, dim=(2, 3), keepdim=True)
                for residual in down_block_res_samples
            ]

        if not return_dict:
            return down_block_res_samples, mid_block_res_sample

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )


# Clearer alias for new code while preserving the original class name and
# checkpoint compatibility.
FoundationNormalControlNetVAEModel = Normal_ControlNetVAEModel
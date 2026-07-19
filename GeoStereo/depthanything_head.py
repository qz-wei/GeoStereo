import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'embed_dim': 1024},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768],  'embed_dim': 768},
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384],   'embed_dim': 384},
    }

    model_configs_dinov2 = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], },
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768],},
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384], },
    }


    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        encoder='vitl',
        tokens_out_dim: int | None = None, 
        do_imagenet_norm: bool = True,
        resize_to: int | None = None,       
        antialias: bool = True,
    ):
        super().__init__()
        from depth_anything.dpt import DepthAnything  

        self.encoder = encoder
        self.do_imagenet_norm = do_imagenet_norm
        self.resize_to = resize_to
        self.antialias = antialias

        depth_anything = DepthAnything(self.model_configs_dinov2[encoder])
        self.depth_anything = depth_anything


        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }


        embed_dim = self.model_configs[encoder]['embed_dim']
        self.tokens_proj = None
        if tokens_out_dim is not None and tokens_out_dim != embed_dim:
            print(f"Projecting tokens from {embed_dim} to {tokens_out_dim} channels for StableNormal compatibility.")
            self.tokens_proj = nn.Conv2d(embed_dim, tokens_out_dim, kernel_size=1, stride=1, padding=0)

        # buffers for normalization
        mean = torch.tensor(self.IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self.IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,                    
        return_depth: bool = True,        
        return_tokens: bool = True,        
    ):

        B, C, H, W = x.shape
        

        patch_size = self.depth_anything.pretrained.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if H % patch_size != 0 or W % patch_size != 0:
            pass

        feats = self.depth_anything.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        out_dict = {}

        if return_tokens:
            last = feats[-1]
            if isinstance(last, (tuple, list)) and len(last) >= 1:
                patch_tokens = last[0]  # [B, N, C]
            else:
                patch_tokens = last

            if patch_tokens.dim() == 3:
                Hp, Wp = x.shape[-2] // patch_size, x.shape[-1] // patch_size
                tokens = patch_tokens.reshape(B, Hp, Wp, patch_tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
            elif patch_tokens.dim() == 4:
                tokens = patch_tokens
            else:
                raise RuntimeError(f"Unexpected patch_tokens shape: {patch_tokens.shape}")

            if self.tokens_proj is not None:
                tokens = self.tokens_proj(tokens)

            out_dict["tokens"] = tokens 


        if return_depth:
            patch_h = x.shape[-2] // patch_size
            patch_w = x.shape[-1] // patch_size

            out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(
                feats, patch_h, patch_w, return_intermediate=True
            )
            out_dict.update({
                "out": out,
                "path_1": path_1,
                "path_2": path_2,
                "path_3": path_3,
                "path_4": path_4,
                "features": feats,
                "disp": disp,
            })

        return out_dict

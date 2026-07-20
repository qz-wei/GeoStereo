import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import json
from torch.utils.data import Dataset
import logging
import cv2
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import logging
from test_normal_329 import StereoCarla,interiorverseDataset,FSDDataset,Structured3DDataset,IRSDataset,HypersimWarpDataset, KenBurnsMonoWarpDataset,NYUv2MonoWarpDataset,DIODEDataset
from PIL import Image
import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
def _to_chw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        if x.shape[0] in (1,3):      # CHW
            return x
        if x.shape[-1] in (1,3):     # HWC
            return x.permute(2,0,1).contiguous()
    return x

def set_global_seed(seed: int = 42):
    """
    固定所有的全局随机种子，确保可复现。
    如果想要每次都不一样，可以传入 seed = int(time.time())
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果用多卡
    
    # 这一步是为了让 cudnn 的卷积算法也固定下来（会牺牲一点点速度，但绝对可复现）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    logging.info(f"Global seed set to {seed}")

# 写一个正确的 worker_init_fn，防止多进程 Dataloader 里的随机数变成一样的剧本
def worker_init_fn(worker_id):
    # 用全局 seed 加上 worker_id，保证每个 worker 的种子不一样，
    # 但每次运行同一个实验时，生成的随机序列又是确定的！
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _pad_or_crop_chw(x: torch.Tensor, th: int, tw: int, pad_val: float = 0.0):
    x = _to_chw(x)
    c,h,w = x.shape
    if h < th or w < tw:
        x = F.pad(x, (0, max(0, tw-w), 0, max(0, th-h)), value=pad_val)
    return x[:, :th, :tw].contiguous()

def make_fixed_hw_collate(target_hw):
    th, tw = target_hw

    def _collate(batch):
        fixed = []
        for sample in batch:
            sample = list(sample)
            # (_, left, right, disp_gt, mask_disp, normal_gt, mask_norm, ...)
            for idx in [1,2,3,4,5,6]:
                if torch.is_tensor(sample[idx]):
                    sample[idx] = _pad_or_crop_chw(sample[idx], th, tw, 0.0)
            fixed.append(tuple(sample))
        return default_collate(fixed)

    return _collate



class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


import numpy as np
import cv2

import numpy as np
import cv2

class StereoAugmentor:
    def __init__(
        self,
        crop_size=(448, 672),
        min_scale=-0.2,
        max_scale=0.5,
        spatial_aug_prob=0.8,
        eps=1e-8,
        min_valid_ratio=0.3,       # 这块裁剪里 mask_disp 有效像素至少占 30%
        max_tries=20,
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.eps = eps
        self.min_valid_ratio = min_valid_ratio
        self.max_tries = max_tries

    def _resize_all(self, img1, img2, disp, normal, mask_disp, mask_normal, scale):
        sx = sy = scale
        img1 = cv2.resize(img1, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

        disp = cv2.resize(disp, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        disp = disp * sx

        normal = cv2.resize(normal, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
        nmag = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (nmag + self.eps)

        mask_disp = cv2.resize(mask_disp, None, fx=sx, fy=sy, interpolation=cv2.INTER_NEAREST)
        mask_normal = cv2.resize(mask_normal, None, fx=sx, fy=sy, interpolation=cv2.INTER_NEAREST)
        return img1, img2, disp, normal, mask_disp, mask_normal

    def _pad_to_min_size(self, img1, img2, disp, normal, mask_disp, mask_normal):
        ch, cw = self.crop_size
        h, w = img1.shape[:2]
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        if pad_h == 0 and pad_w == 0:
            return img1, img2, disp, normal, mask_disp, mask_normal

        # 只在底部/右侧 pad（简单稳定）
        img1 = cv2.copyMakeBorder(img1, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        img2 = cv2.copyMakeBorder(img2, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        disp = cv2.copyMakeBorder(disp, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        normal = cv2.copyMakeBorder(normal, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask_disp = cv2.copyMakeBorder(mask_disp, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask_normal = cv2.copyMakeBorder(mask_normal, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return img1, img2, disp, normal, mask_disp, mask_normal

    def _crop_prefer_valid(self, img1, img2, disp, normal, mask_disp, mask_normal):
        ch, cw = self.crop_size
        h, w = img1.shape[:2]
        if h == ch and w == cw:
            return img1, img2, disp, normal, mask_disp, mask_normal

        # 保底：随机裁一次
        def do_crop(y0, x0):
            y1, x1 = y0 + ch, x0 + cw
            return (
                img1[y0:y1, x0:x1],
                img2[y0:y1, x0:x1],
                disp[y0:y1, x0:x1],
                normal[y0:y1, x0:x1],
                mask_disp[y0:y1, x0:x1],
                mask_normal[y0:y1, x0:x1],
            )

        # 尝试找 mask_disp 有效比例高的位置
        best = None
        best_ratio = -1.0
        for _ in range(self.max_tries):
            y0 = np.random.randint(0, h - ch + 1) if h > ch else 0
            x0 = np.random.randint(0, w - cw + 1) if w > cw else 0
            md = mask_disp[y0:y0+ch, x0:x0+cw]
            ratio = float((md > 0).mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best = (y0, x0)
            if ratio >= self.min_valid_ratio:
                return do_crop(y0, x0)

        # 找不到就用最佳的那个（至少比纯随机强）
        y0, x0 = best
        return do_crop(y0, x0)

    def __call__(self, img1, img2, disp, normal, mask_disp, mask_normal):
        if np.random.rand() < self.spatial_aug_prob:
            ht, wd = img1.shape[:2]
            ch, cw = self.crop_size

            # ✅ 关键：缩放下限，保证缩放后至少能 crop 到 crop_size
            min_scale_needed = max((ch + 1) / float(ht), (cw + 1) / float(wd))
            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale = max(scale, min_scale_needed)

            img1, img2, disp, normal, mask_disp, mask_normal = self._resize_all(
                img1, img2, disp, normal, mask_disp, mask_normal, scale
            )

        # ✅ 裁剪前确保尺寸足够
        img1, img2, disp, normal, mask_disp, mask_normal = self._pad_to_min_size(
            img1, img2, disp, normal, mask_disp, mask_normal
        )

        # ✅ 裁剪回固定大小，并尽量避开全 mask 区域
        img1, img2, disp, normal, mask_disp, mask_normal = self._crop_prefer_valid(
            img1, img2, disp, normal, mask_disp, mask_normal
        )

        return (
            np.ascontiguousarray(img1),
            np.ascontiguousarray(img2),
            np.ascontiguousarray(disp),
            np.ascontiguousarray(normal),
            np.ascontiguousarray(mask_disp),
            np.ascontiguousarray(mask_normal),
        )
def fetch_dataloader(args, rank=None, world_size=None, mode='found'):
    """
    Create the data loader for the corresponding training set.
    
    Args:
        args: Training arguments
        rank: Current process rank for DDP (None for single GPU)
        world_size: Total number of processes for DDP (None for single GPU)
    
    Returns:
        DataLoader with DistributedSampler if DDP is enabled
    """
    
    # Setup augmentation parameters
    if mode == 'found':
        image_size = args.image_size_found
        aug_params = {
            'crop_size': args.image_size_found, 
            'min_scale': args.spatial_scale[0], 
            'max_scale': args.spatial_scale[1], 
            'do_flip': False, 
            'yjitter': not args.noyjitter
        }
    else:
        image_size = args.image_size_diff
        aug_params = {
            'crop_size': args.image_size_diff, 
            'min_scale': args.spatial_scale[0], 
            'max_scale': args.spatial_scale[1], 
            'do_flip': False, 
            'yjitter': not args.noyjitter
        }
    
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None

    # ---- 修改点1：除 mixstereo 外，尽量强制 is_test=True（类不支持就自动回退不传）----
    force_is_test = (args.train_datasets != 'mixstereo')

    def build(ds_cls, *a, **kw):
        if force_is_test and ("is_test" not in kw):
            try:
                return ds_cls(*a, is_test=True, **kw)
            except TypeError:
                return ds_cls(*a, **kw)
        return ds_cls(*a, **kw)

    # Dataset selection logic
    if args.train_datasets == 'sceneflow':
        pass

    elif args.train_datasets == 'mixstereo':
        # mix 不变
        a3d_ken=KenBurnsMonoWarpDataset(
            txt_list='/Node09_nvme/weiqizhe/eval/3d.txt',
            image_size=image_size,
        )
        logging.info(f"Adding {len(a3d_ken)} samples from 3d_ken")

        Inter=interiorverseDataset(
                jsonl_file = "/gpfs/essfs/public_data/test/interiorverse/interiorverse_data.jsonl",
                data_root = "/gpfs/essfs/public_data/test/interiorverse/interiorverse" ,
                image_size=image_size,
        )
        logging.info(f"Adding {len(Inter)} samples from Inter")

        struct3d=Structured3DDataset(
                jsonl_file = "/gpfs/essfs/public_data/test/Structured3D_perspective_full/structured3d_train_relative.jsonl",
                data_root = "/gpfs/essfs/public_data/test/Structured3D_perspective_full",
                image_size=image_size,
        )
        logging.info(f"Adding {len(struct3d)} samples from struct3d")
        IRS=IRSDataset(
                            data_root="/gpfs/essfs/public_data/test/IRS/IRS",
                jsonl_file="/Node09_nvme/weiqizhe/train.dedup.list",
                image_size=image_size,
            )
        hypersim=HypersimWarpDataset(
            jsonl_file='/gpfs/essfs/public_data/test/hypersim/hypersim.jsonl',
            image_size=image_size,
        )
        nyu= NYUv2MonoWarpDataset(
        data_root="/Node09_nvme/weiqizhe/eval/nyuv2/data/unzip",
        split="train",
        image_size=image_size,
        is_test=False
        )

        fsd = FSDDataset(
            data_root='/gpfs/essfs/public_data/test/FSD/FSD/FSD',
            jsonl_file='/Node09_nvme/weiqizhe/eval/FSD.jsonl',
            image_size=image_size,

        )
        logging.info(f"Adding {len(fsd)} samples from fsd")
        stereocarla = build(StereoCarla,
            txt_file = "/Node09_nvme/weiqizhe/eval/carla.txt",
            data_root = "/gpfs/essfs/public_data/test/StereoCarla",
            image_size=image_size,
        )
        logging.info(f"Adding {len(stereocarla)} samples from stereocarla")


        diode1 =DIODEDataset(
            data_root='/gpfs/essfs/public_data/test/diode/',
            csv_file = '/gpfs/essfs/public_data/test/diode/data_list/train_indoors.csv',
            image_size=image_size,
        )

        diode2 =DIODEDataset(
            data_root='/gpfs/essfs/public_data/test/diode/',
            csv_file = '/gpfs/essfs/public_data/test/diode/data_list/train_outdoor.csv',
            image_size=image_size,
        )


        diode3 =DIODEDataset(
            data_root='/gpfs/essfs/public_data/test/diode/',
            csv_file = '/gpfs/essfs/public_data/test/diode/data_list/val_indoors.csv',
            image_size=image_size,
        )

        diode4 =DIODEDataset(
            data_root='/gpfs/essfs/public_data/test/diode/',
            csv_file = '/gpfs/essfs/public_data/test/diode/data_list/val_outdoor.csv',
            image_size=image_size,
        )

        new_dataset= diode1 + diode2 +nyu + IRS+ struct3d+ Inter + hypersim + a3d_ken  # fsd+ stereocarla +  diode1 + diode2 + IRS+ struct3d+ Inter + hypersim + a3d_ken

        logging.info(f"Adding {len(new_dataset)} samples from mixstereo Dataset")

    elif args.train_datasets == 'struct3d':
        struct3d=build(Structured3DDataset,
                jsonl_file = "/gpfs/essfs/public_data/test/Structured3D_perspective_full/structured3d_test_relative.jsonl",
                data_root = "/gpfs/essfs/public_data/test/Structured3D_perspective_full",
                image_size=image_size,
        )
        new_dataset=struct3d


    elif args.train_datasets == 'irs':
        IRS=build(IRSDataset,
                data_root="/gpfs/essfs/public_data/test/IRS/IRS",
                jsonl_file="/Node09_nvme/weiqizhe/train.dedup.list",
                image_size=image_size,
            )
        new_dataset=IRS
        logging.info(f"Adding {len(new_dataset)} samples from irs Dataset")

    elif args.train_datasets == 'interiorverse':
        Inter=build(interiorverseDataset,
                jsonl_file = "/gpfs/essfs/public_data/test/interiorverse/interiorverse_data_test.jsonl",
                data_root = "/gpfs/essfs/public_data/test/interiorverse/interiorverse" ,
                image_size=image_size,
        )
        new_dataset=Inter
        logging.info(f"Adding {len(new_dataset)} samples from interiorverse Dataset")

    elif args.train_datasets == 'hypersim':
        hypersim=build(HypersimWarpDataset,
            jsonl_file='/gpfs/essfs/public_data/test/hypersim/hypersim_test.jsonl',
            image_size=image_size, 
        )
        new_dataset=hypersim
        logging.info(f"Adding {len(new_dataset)} samples from hypersim Dataset")        

    elif args.train_datasets == 'stereocarla':  
        stereocarla = build(StereoCarla,
            txt_file = "/Node09_nvme/weiqizhe/eval/carla.txt",
            data_root = "/gpfs/essfs/public_data/test/StereoCarla",
            image_size=image_size,
        )
        new_dataset=stereocarla
        logging.info(f"Adding {len(new_dataset)} samples from Stereocarla Dataset") 

    elif args.train_datasets == 'irs_test':
        IRS=build(IRSDataset,
                data_root="/gpfs/essfs/public_data/test/IRS/IRS/data",
                jsonl_file="/Node09_nvme/weiqizhe/eval/IRS/sample_50.list",
                image_size=image_size,
            )

        new_dataset=IRS
        logging.info(f"Adding {len(new_dataset)} samples from irs_test Dataset")

    elif args.train_datasets == 'kenburns':
        a3d_ken=build(KenBurnsMonoWarpDataset,
            txt_list='/Node09_nvme/weiqizhe/eval/3d_test.txt',
            image_size=image_size,
            use_warp_right=True
        )
        new_dataset=a3d_ken
        logging.info(f"Adding {len(new_dataset)} samples from 3d_ken Dataset")

    elif args.train_datasets == 'nyu':
        nyu= NYUv2MonoWarpDataset(
        data_root="/Node09_nvme/weiqizhe/eval/nyuv2/data/unzip",
        split="test",
        image_size=image_size,
        is_test=True

        )
        new_dataset = nyu
    elif args.train_datasets == 'fsd':
        fsd = FSDDataset(
            data_root='/gpfs/essfs/public_data/test/FSD/FSD/FSD',
            jsonl_file='/Node09_nvme/weiqizhe/eval/FSD.jsonl',
            image_size=image_size,
            is_test=False

        )
        new_dataset =fsd
    elif args.train_datasets == 'diode':
        diode =DIODEDataset(
            data_root='/gpfs/essfs/public_data/test/diode/',
            csv_file = '/gpfs/essfs/public_data/test/diode/data_list/train_indoors.csv',
            image_size=image_size,
        )
        new_dataset = diode
    else:
        raise ValueError(f"Unknown dataset: {args.train_datasets}")

    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    
    logging.info(f'Training with {len(train_dataset)} image pairs')
    
    # Create DataLoader with DDP support
    use_ddp = (rank is not None and world_size is not None and world_size > 1)

    # ---- 修改点2：collate_fn 仅 mixstereo 使用；其他不传 collate_fn ----
    collate_fn = make_fixed_hw_collate(tuple(image_size)) if args.train_datasets == 'mixstereo' else None

    def _make_loader(dataset, **kwargs):
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        kwargs["worker_init_fn"] = worker_init_fn    
        return data.DataLoader(dataset, **kwargs)

    if use_ddp:
        # Use DistributedSampler for DDP
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        if mode == 'found':
            train_loader = _make_loader(
                train_dataset, 
                batch_size=args.batch_size_found,  # batch_size per GPU
                sampler=sampler,
                pin_memory=True, 
                num_workers=8, 
                drop_last=True,
            )
        else:
            train_loader = _make_loader(
                train_dataset, 
                batch_size=args.batch_size_diff,  # batch_size per GPU
                sampler=sampler,
                pin_memory=True, 
                num_workers=8, 
                drop_last=True,
            )
        
        if rank == 0:
            logging.info(f'Using DistributedSampler with {world_size} GPUs')
    else:
        # Single GPU or DataParallel mode
        if mode == 'found':
            train_loader = _make_loader(
                train_dataset, 
                batch_size=args.batch_size_found, 
                pin_memory=True, 
                shuffle=True, 
                num_workers=8, 
                drop_last=True,
            )
        else:
            train_loader = _make_loader(
                train_dataset, 
                batch_size=args.batch_size_diff, 
                pin_memory=True, 
                shuffle=True, 
                num_workers=8, 
                drop_last=True,
            )
        
        logging.info('Using single GPU or DataParallel mode')

    return train_loader
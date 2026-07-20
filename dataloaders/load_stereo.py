import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
import json
import h5py
from .mono import WarpDataset
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import torch
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import pandas as pd
import h5py
#################
# Dataset Mixer
#################
import numpy as np
import logging
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class MixedDataLoaderN:
    def __init__(self, loaders, splits, seed=42):
        """
        loaders: list of dataloader
        splits:  list of desired mixing ratio, e.g. [8, 1, 1]
        seed:    base seed for deterministic shared schedule across ranks
        """
        assert len(loaders) == len(splits), "loaders 和 splits 长度必须一致"
        assert len(loaders) > 0, "至少要有一个 loader"
        assert all(s > 0 for s in splits), "splits 里每个值都必须 > 0"

        self.loaders = loaders
        self.splits = splits
        self.seed = int(seed)
        self.epoch = 0

        self.counts = self._compute_counts()
        self.schedule = None
        self.loader_iters = None
        self.idx = 0

    def _compute_counts(self):
        """
        counts[i] = alpha * splits[i]
        alpha 取能满足所有 loader 不越界的最大值
        """
        sizes = [len(loader) for loader in self.loaders]
        alpha = min(size / split for size, split in zip(sizes, self.splits))
        counts = [int(alpha * split) for split in self.splits]

        print("Loader sizes:", sizes)
        print("Mix splits:", self.splits)
        print("Actual counts:", counts)

        return counts

    def set_epoch(self, epoch: int):
        """
        Make schedule deterministic and identical across ranks for each epoch.
        Also forward epoch to distributed samplers when possible.
        """
        self.epoch = int(epoch)

        for loader in self.loaders:
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            elif hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            elif hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "sampler") \
                 and hasattr(loader.batch_sampler.sampler, "set_epoch"):
                loader.batch_sampler.sampler.set_epoch(epoch)

    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]

        schedule = []
        for i, c in enumerate(self.counts):
            schedule.extend([i] * c)

        # IMPORTANT: all ranks get the same schedule for the same epoch
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(schedule)

        self.schedule = schedule
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.schedule):
            raise StopIteration

        loader_id = self.schedule[self.idx]
        self.idx += 1
        return next(self.loader_iters[loader_id])

    def __len__(self):
        return sum(self.counts)


class SynchronizedTransform_Hyper:
    def __init__(self, H=576, W=768):
        self.resize_rgb = transforms.Resize((H, W), interpolation=Image.BILINEAR)
        self.resize_depth = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.resize_normal = transforms.Resize((H, W), interpolation=Image.NEAREST)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, left_rgb_image, right_rgb_image, depth_image, normal_image=None):
        left_rgb_image = self.resize_rgb(left_rgb_image)
        right_rgb_image = self.resize_rgb(right_rgb_image)
        depth_image = self.resize_depth(depth_image)

        if normal_image is not None:
            normal_image = self.resize_normal(normal_image)

        left_rgb_tensor = self.to_tensor(left_rgb_image)
        right_rgb_tensor = self.to_tensor(right_rgb_image)
        depth_tensor = self.to_tensor(depth_image)

        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
            return left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor

        return left_rgb_tensor, right_rgb_tensor, depth_tensor



class HypersimStereoJSONL(Dataset):
    def __init__(
        self,
        jsonl_path= '/file_system/goosefsx/whu/users/hong.li/qizhe/process_datasets/hypersim/hypersim_new_rgb_normal_depth.jsonl',
        transform=True,
        near_plane=1e-5,
        far_plane=65.0,
        align_cam_normal=True,
    ):
        self.jsonl_path = jsonl_path
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.align_cam_normal = align_cam_normal

        self.samples = self._load_jsonl()

        self.transform = SynchronizedTransform_Hyper() if transform else None

        # mono warper for online right-view generation
        self.warper = WarpDataset()
        self.warper.disable_background = False
        self.warper.disable_sharpening = False

    def _load_jsonl(self):
        samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                rgb_hdf5 = item["rgb_hdf5"]
                depth_hdf5 = item["depth_hdf5"]
                normal_hdf5 = item["normal_hdf5"]
                meta = item.get("meta", {})

                if not (os.path.exists(rgb_hdf5) and os.path.exists(depth_hdf5) and os.path.exists(normal_hdf5)):
                    continue

                samples.append(
                    {
                        "rgb_hdf5": rgb_hdf5,
                        "depth_hdf5": depth_hdf5,
                        "normal_hdf5": normal_hdf5,
                        "meta": meta,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    #####################
    # Generic HDF5 utils
    #####################

    def _find_first_dataset(self, group):
        if isinstance(group, h5py.Dataset):
            return group[()]
        for key in group.keys():
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            if isinstance(obj, h5py.Group):
                arr = self._find_first_dataset(obj)
                if arr is not None:
                    return arr
        return None

    def _read_hdf5_any(self, path, candidate_keys=None):
        if candidate_keys is None:
            candidate_keys = []

        with h5py.File(path, "r") as f:
            for k in candidate_keys:
                if k in f:
                    return f[k][()]
            arr = self._find_first_dataset(f)

        if arr is None:
            raise ValueError(f"No dataset found in HDF5: {path}")
        return arr

    #####################
    # Hypersim readers
    #####################
    def read_hypersim_color_u8(self, path):
        arr = self._read_hdf5_any(path, candidate_keys=["dataset", "rgb", "color"])
        arr = np.array(arr)

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if arr.ndim != 3:
            raise ValueError(f"Unsupported RGB shape {arr.shape} in {path}")

        if arr.shape[2] == 4:
            arr = arr[..., :3]

        orig_dtype = arr.dtype
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # uint8 / integer-like image
        if not np.issubdtype(orig_dtype, np.floating):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        # float image
        arr = arr.astype(np.float32)

        # case 1: already [0,1]
        if arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-6:
            # linear -> display-ish
            arr = np.clip(arr, 0.0, 1.0)
            arr = arr ** (1.0 / 2.2)
            arr = (arr * 255.0).astype(np.uint8)
            return arr

        # case 2: HDR / linear radiance
        # use deterministic tone mapping instead of per-image percentile stretching
        arr = np.clip(arr, 0.0, None)
        arr = arr / (1.0 + arr)          # Reinhard tone mapping
        arr = arr ** (1.0 / 2.2)         # gamma
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        return arr

    def read_hypersim_depth_m(self, path):
        arr = self._read_hdf5_any(path, candidate_keys=["dataset", "depth"])
        arr = np.array(arr)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]

        if arr.ndim != 2:
            raise ValueError(f"Unsupported depth shape {arr.shape} in {path}")

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr

    def read_hypersim_normal_cam_m11(self, path):
        """
        Returns float32 normal in [-1, 1], shape [H, W, 3]
        """
        arr = self._read_hdf5_any(path, candidate_keys=["dataset", "normal"])
        arr = np.array(arr)

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3:
            raise ValueError(f"Unsupported normal shape {arr.shape} in {path}")

        if arr.shape[2] == 4:
            arr = arr[..., :3]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # If normals are stored in [0,1], map to [-1,1]
        if arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-6:
            arr = arr * 2.0 - 1.0

        # If somehow stored in [0,255], map to [-1,1]
        elif arr.min() >= 0.0 and arr.max() <= 255.0 + 1e-6:
            arr = arr / 255.0 * 2.0 - 1.0

        # normalize
        norm = np.linalg.norm(arr, axis=2, keepdims=True)
        arr = arr / (norm + 1e-6)
        return arr.astype(np.float32)

    def _apply_normal_flips(self, normal):
        """
        Match the convention you were using before:
        - flip x if hypersim x-axis is opposite to target convention
        - keep existing y/z flips
        """
        normal = normal.copy()
        
        normal[:, :, 1:] *= -1.0  # existing y/z flip
        return normal

    #####################
    # Geometry helpers
    #####################

    def creat_uv_mesh(self, H, W):
        y, x = np.meshgrid(
            np.arange(0, H, dtype=np.float64),
            np.arange(0, W, dtype=np.float64),
            indexing="ij"
        )
        meshgrid = np.stack((x, y))
        ones = np.ones((1, H * W), dtype=np.float64)
        xy = meshgrid.reshape(2, -1)
        return np.concatenate([xy, ones], axis=0)

    def align_normals(self, normal, depth, K, H, W):
        """
        Orientation of surface normals in hypersim is not always consistent
        """
        K = np.array([
            [K[0],    0, K[2]],
            [   0, K[1], K[3]],
            [   0,    0,    1]
        ])
        inv_K = np.linalg.inv(K)

        xy = self.creat_uv_mesh(H, W)
        points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
        points = depth * points
        points = points.transpose((1, 2, 0))

        orient_mask = np.sum(normal * points, axis=2) > 0
        normal[orient_mask] *= -1
        return normal

    def _process_depth_and_generate_stereo(self, img_left_rgb_u8: np.ndarray, depth_m: np.ndarray):
        """
        img_left_rgb_u8: [H,W,3] uint8 (RGB)
        depth_m: [H,W] float32 meters
        return:
          img_right_rgb_u8: [H,W,3] uint8 (RGB)
          final_disp_map: [H,W] float32
        """
        img_left = img_left_rgb_u8
        h, w = img_left.shape[:2]
        max_disp_padding = int(self.warper.max_disparity)

        self.warper.feed_height = h
        self.warper.feed_width = w
        self.warper.process_width = w + max_disp_padding
        self.warper.xs, self.warper.ys = np.meshgrid(
            np.arange(self.warper.process_width),
            np.arange(self.warper.feed_height)
        )

        mask_valid = (depth_m > 1e-6) & (depth_m < 1e5) & np.isfinite(depth_m)

        raw_disp = np.zeros_like(depth_m, dtype=np.float32)

        if mask_valid.sum() > 0:
            d = depth_m[mask_valid]
            d_min = np.percentile(d, 2)
            d_max = np.percentile(d, 98)
            if d_max <= d_min:
                d_max = d_min + 1e-6

            depth_norm = np.zeros_like(depth_m, dtype=np.float32)
            depth_norm[mask_valid] = (depth_m[mask_valid] - d_min) / (d_max - d_min)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)

            # near large, far small
            raw_disp = 1.0 - depth_norm

        raw_disp_padded = np.pad(raw_disp, ((0, 0), (0, max_disp_padding)), mode="reflect")
        img_left_padded = np.pad(img_left, ((0, 0), (0, max_disp_padding), (0, 0)), mode="reflect")

        final_disp_map_padded = self.warper.process_disparity(raw_disp_padded)

        img_right_padded = self.warper.project_image(
            img_left_padded,
            final_disp_map_padded,
            background_image=img_left_padded
        )

        img_right = img_right_padded[:, :w, :]
        final_disp_map = final_disp_map_padded[:, :w].astype(np.float32)
        return img_right.astype(np.uint8), final_disp_map

    #####################
    # Main
    #####################

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb_h5 = sample["rgb_hdf5"]
        dep_h5 = sample["depth_hdf5"]
        nor_h5 = sample["normal_hdf5"]
        meta = sample.get("meta", {})

        left_rgb = self.read_hypersim_color_u8(rgb_h5)          # [H,W,3] uint8
        depth_m = self.read_hypersim_depth_m(dep_h5)            # [H,W] float32 meters
        normal = self.read_hypersim_normal_cam_m11(nor_h5)      # [H,W,3] float32 in [-1,1]

        normal = self._apply_normal_flips(normal)

        #normal = np.ascontiguousarray(np.fliplr(normal))

        # 镜像后，法向量 x 分量要取反
        #normal[:, :, 0] *= -1.0
        
        

        if self.align_cam_normal:
            H, W = normal.shape[:2]
            normal = self.align_normals(
                normal,
                depth_m,
                [886.81, 886.81, W / 2, H / 2],
                H,
                W,
            ) * -1.0

        right_rgb, disp = self._process_depth_and_generate_stereo(left_rgb, depth_m)

        # convert to PIL for shared transform path
        left_rgb_image = Image.fromarray(left_rgb)
        right_rgb_image = Image.fromarray(right_rgb)
        depth_image = Image.fromarray(depth_m.astype(np.float32))

        # [-1,1] -> [0,255] for transform path
        normal_u8 = ((np.clip(normal, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        normal_image = Image.fromarray(normal_u8)

        # transform
        if self.transform is not None:
            left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor = self.transform(
                left_rgb_image, right_rgb_image, depth_image, normal_image
            )
        else:
            left_rgb_tensor = transforms.ToTensor()(left_rgb_image)
            right_rgb_tensor = transforms.ToTensor()(right_rgb_image)
            depth_tensor = transforms.ToTensor()(depth_image)
            normal_tensor = transforms.ToTensor()(normal_image)

        # valid depth mask
        valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)

        # RGB -> [-1,1]
        left_rgb_tensor = left_rgb_tensor * 2.0 - 1.0
        right_rgb_tensor = right_rgb_tensor * 2.0 - 1.0

        # depth processing
        if valid_depth_mask.any():
            flat_depth = depth_tensor[valid_depth_mask].flatten().float()
            min_depth = torch.quantile(flat_depth, 0.02)
            max_depth = torch.quantile(flat_depth, 0.98)
            if min_depth == max_depth:
                depth_tensor = torch.zeros_like(depth_tensor)
                metric_tensor = torch.zeros_like(depth_tensor)
                valid_depth_mask = torch.zeros_like(depth_tensor).bool()
            else:
                depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth)
                depth_tensor[~valid_depth_mask] = max_depth
                metric_tensor = depth_tensor.clone()
                depth_tensor = torch.clamp(
                    (((depth_tensor - min_depth) / (max_depth - min_depth)) * 2.0) - 1.0,
                    -1,
                    1,
                )
        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)

        depth_tensor = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze()

        # normal -> [-1,1]
        normal_tensor = normal_tensor * 2.0 - 1.0
        normal_tensor = torch.nn.functional.normalize(normal_tensor, p=2, dim=0)
        normal_tensor[0, ~valid_depth_mask.squeeze()] = 0
        normal_tensor[1, ~valid_depth_mask.squeeze()] = 0
        normal_tensor[2, ~valid_depth_mask.squeeze()] = 0
        normal_tensor[0] = -normal_tensor[0]

        return {
            "left_rgb": left_rgb_tensor,
            "right_rgb": right_rgb_tensor,
            "depth": depth_tensor,
            "metric": metric_tensor,
            "normals": normal_tensor,
            "val_mask": valid_depth_mask,
            "domain": "indoor",
            "left_rgb_path": rgb_h5,
            "right_rgb_path": rgb_h5.replace(".color.hdf5", ".color_right_warped.hdf5"),
            "depth_path": dep_h5,
            "normal_path": nor_h5,
            "meta": meta,
            "mono_disp": torch.from_numpy(disp).unsqueeze(0),  # optional debug
        }


class SynchronizedTransform_VKITTI_Stereo:
    TARGET_H = 576
    TARGET_W = 768

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def resize_keep_aspect_to_height(self, image, interpolation):
        w, h = image.size
        new_w = int(round(w * self.TARGET_H / h))
        return TF.resize(image, (self.TARGET_H, new_w), interpolation=interpolation)

    def center_crop_width(self, image):
        w, h = image.size
        if h != self.TARGET_H:
            raise ValueError(f"Unexpected VKITTI height after resize: {h}")
        if w < self.TARGET_W:
            raise ValueError(f"Unexpected VKITTI width after resize: {w} < {self.TARGET_W}")

        left = (w - self.TARGET_W) // 2
        return TF.crop(image, top=0, left=left, height=self.TARGET_H, width=self.TARGET_W)

    def __call__(self, left_rgb_image, right_rgb_image, depth_image, normal_image=None):
        left_rgb_image = self.resize_keep_aspect_to_height(left_rgb_image, InterpolationMode.BILINEAR)
        right_rgb_image = self.resize_keep_aspect_to_height(right_rgb_image, InterpolationMode.BILINEAR)
        depth_image = self.resize_keep_aspect_to_height(depth_image, InterpolationMode.NEAREST)

        if normal_image is not None:
            normal_image = self.resize_keep_aspect_to_height(normal_image, InterpolationMode.NEAREST)

        left_rgb_image = self.center_crop_width(left_rgb_image)
        right_rgb_image = self.center_crop_width(right_rgb_image)
        depth_image = self.center_crop_width(depth_image)

        if normal_image is not None:
            normal_image = self.center_crop_width(normal_image)

        left_rgb_tensor = self.to_tensor(left_rgb_image)
        right_rgb_tensor = self.to_tensor(right_rgb_image)
        depth_tensor = self.to_tensor(depth_image)

        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
            return left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor

        return left_rgb_tensor, right_rgb_tensor, depth_tensor


class SynchronizedTransform_VKITTI_Stereo1:
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 960

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def get_random_kitti_crop_params(self, image):
        """
        高度：保留下方 352，裁掉上方天空
        宽度：随机左右裁剪到 960
        """
        w, h = image.size

        if h < self.KB_CROP_HEIGHT:
            raise ValueError(
                f"Image height {h} < KB_CROP_HEIGHT {self.KB_CROP_HEIGHT}"
            )
        if w < self.KB_CROP_WIDTH:
            raise ValueError(
                f"Image width {w} < KB_CROP_WIDTH {self.KB_CROP_WIDTH}"
            )

        top = h - self.KB_CROP_HEIGHT
        left = random.randint(0, w - self.KB_CROP_WIDTH)

        return top, left

    def crop_with_params(self, image, top, left):
        return TF.crop(
            image,
            top=top,
            left=left,
            height=self.KB_CROP_HEIGHT,
            width=self.KB_CROP_WIDTH,
        )

    def __call__(self, left_rgb_image, right_rgb_image, depth_image, normal_image=None):
        # 只随机一次，所有图共用同一个 crop 参数，保证对齐
        top, left = self.get_random_kitti_crop_params(left_rgb_image)

        left_rgb_image = self.crop_with_params(left_rgb_image, top, left)
        right_rgb_image = self.crop_with_params(right_rgb_image, top, left)
        depth_image = self.crop_with_params(depth_image, top, left)

        if normal_image is not None:
            normal_image = self.crop_with_params(normal_image, top, left)

        left_rgb_tensor = self.to_tensor(left_rgb_image)
        right_rgb_tensor = self.to_tensor(right_rgb_image)
        depth_tensor = self.to_tensor(depth_image)

        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
            return left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor

        return left_rgb_tensor, right_rgb_tensor, depth_tensor




class VirtualKITTI2Stereo(Dataset):
    """
    One sample corresponds to the same scene/weather/frame from:
      - left_rgb  = Camera_0
      - right_rgb = Camera_1
      - depth     = Camera_0
      - normals   = Camera_0
    """

    def __init__(self, root_dir='/file_system/goosefsx/public_data/DepthAnything3/qizhe/datasets/virtualkitti2', transform=None, near_plane=1e-5, far_plane=80.0):
        self.root_dir = root_dir
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.pairs = self._find_pairs()
        self.transform = SynchronizedTransform_VKITTI_Stereo1() if transform else None

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = [
            "15-deg-left", "15-deg-right",
            "30-deg-left", "30-deg-right",
            "clone", "morning", "fog", "rain", "sunset", "overcast"
        ]

        pairs = []
        for scene in scenes:
            for weather in weather_conditions:
                frames_dir = os.path.join(self.root_dir, scene, weather, "frames")

                left_rgb_dir = os.path.join(frames_dir, "rgb", "Camera_0")
                right_rgb_dir = os.path.join(frames_dir, "rgb", "Camera_1")
                left_depth_dir = os.path.join(frames_dir, "depth", "Camera_0")
                left_normal_dir = os.path.join(frames_dir, "normal", "Camera_0")

                if not (
                    os.path.exists(left_rgb_dir)
                    and os.path.exists(right_rgb_dir)
                    and os.path.exists(left_depth_dir)
                    and os.path.exists(left_normal_dir)
                ):
                    continue

                left_rgb_files = sorted(
                    [f for f in os.listdir(left_rgb_dir) if f.endswith(".jpg") and f.startswith("rgb_")]
                )

                for left_rgb_file in left_rgb_files:
                    suffix = left_rgb_file[3:]  # strip "rgb", e.g. "_00001.jpg"

                    right_rgb_file = "rgb" + suffix
                    depth_file = "depth" + suffix.replace(".jpg", ".png")
                    normal_file = "normal" + suffix.replace(".jpg", ".png")

                    left_rgb_path = os.path.join(left_rgb_dir, left_rgb_file)
                    right_rgb_path = os.path.join(right_rgb_dir, right_rgb_file)
                    depth_path = os.path.join(left_depth_dir, depth_file)
                    normal_path = os.path.join(left_normal_dir, normal_file)

                    if (
                        os.path.exists(left_rgb_path)
                        and os.path.exists(right_rgb_path)
                        and os.path.exists(depth_path)
                        and os.path.exists(normal_path)
                    ):
                        pairs.append({
                            "left_rgb_path": left_rgb_path,
                            "right_rgb_path": right_rgb_path,
                            "depth_path": depth_path,
                            "normal_path": normal_path,
                            "scene": scene,
                            "weather": weather,
                            "frame": suffix,
                        })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _read_vkitti_normal_png16(self, path):
        """
        Read VKITTI normal saved as 16-bit PNG:
            est_normal_16bit = ((normal + 1) * 32767.5).astype(np.uint16)
            then saved by cv2 in BGR order
        Returns:
            float32 [H, W, 3] in [-1, 1]
        """
        normal = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if normal is None:
            raise FileNotFoundError(f"Normal not found: {path}")

        if normal.ndim != 3:
            raise ValueError(f"Unexpected normal shape {normal.shape} in {path}")

        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32)
        normal = normal / 32767.5 - 1.0
        normal = np.clip(normal, -1.0, 1.0)

        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-6)
        normal[:, :, 0] *= -1.0   # flip x
        return normal.astype(np.float32)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # RGB
        left_rgb_image = Image.open(pair["left_rgb_path"]).convert("RGB")
        right_rgb_image = Image.open(pair["right_rgb_path"]).convert("RGB")

        # Depth from Camera_0
        depth_image = cv2.imread(pair["depth_path"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32) / 100.0  # cm -> m
        depth_image = Image.fromarray(depth_image)

        # Normal from Camera_0 (16-bit PNG -> [-1,1] -> uint8 for transform path)
        normal = self._read_vkitti_normal_png16(pair["normal_path"])
        normal_u8 = ((np.clip(normal, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
        normal_image = Image.fromarray(normal_u8)

        # Transform
        if self.transform is not None:
            left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor = self.transform(
                left_rgb_image, right_rgb_image, depth_image, normal_image
            )
        else:
            left_rgb_tensor = transforms.ToTensor()(left_rgb_image)
            right_rgb_tensor = transforms.ToTensor()(right_rgb_image)
            depth_tensor = transforms.ToTensor()(depth_image)
            normal_tensor = transforms.ToTensor()(normal_image)

        # Depth mask
        valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)

        # Process RGB
        left_rgb_tensor = left_rgb_tensor * 2.0 - 1.0
        right_rgb_tensor = right_rgb_tensor * 2.0 - 1.0

        # Process depth
        if valid_depth_mask.any():
            flat_depth = depth_tensor[valid_depth_mask].flatten().float()
            min_depth = torch.quantile(flat_depth, 0.02)
            max_depth = torch.quantile(flat_depth, 0.98)
            if min_depth == max_depth:
                depth_tensor = torch.zeros_like(depth_tensor)
                metric_tensor = torch.zeros_like(depth_tensor)
                valid_depth_mask = torch.zeros_like(depth_tensor).bool()
            else:
                depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth)
                depth_tensor[~valid_depth_mask] = max_depth
                metric_tensor = depth_tensor.clone()
                depth_tensor = torch.clamp(
                    (((depth_tensor - min_depth) / (max_depth - min_depth)) * 2.0) - 1.0,
                    -1,
                    1,
                )
        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)

        depth_tensor = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze()

        # Process normals
        normal_tensor = normal_tensor * 2.0 - 1.0
        normal_tensor = torch.nn.functional.normalize(normal_tensor, p=2, dim=0)
        normal_tensor[0, ~valid_depth_mask.squeeze()] = 0
        normal_tensor[1, ~valid_depth_mask.squeeze()] = 0
        normal_tensor[2, ~valid_depth_mask.squeeze()] = 0

        return {
            "left_rgb": left_rgb_tensor,
            "right_rgb": right_rgb_tensor,
            "depth": depth_tensor,
            "metric": metric_tensor,
            "normals": normal_tensor,
            "val_mask": valid_depth_mask,
            "domain": "outdoor",
            "left_rgb_path": pair["left_rgb_path"],
            "right_rgb_path": pair["right_rgb_path"],
            "depth_path": pair["depth_path"],
            "normal_path": pair["normal_path"],
        }


class SynchronizedTransform_IRS:
    def __init__(
        self,
        target_h=540,
        target_w=960,
        keep_aspect=False,
        crop_width=False,
    ):
        self.target_h = target_h
        self.target_w = target_w
        self.keep_aspect = keep_aspect
        self.crop_width = crop_width
        self.to_tensor = transforms.ToTensor()

    def resize(self, image, interpolation):
        if self.keep_aspect:
            w, h = image.size
            new_w = int(round(w * self.target_h / h))
            return TF.resize(image, (self.target_h, new_w), interpolation=interpolation)
        else:
            return TF.resize(image, (self.target_h, self.target_w), interpolation=interpolation)

    def maybe_center_crop_width(self, image):
        if not self.crop_width:
            return image

        w, h = image.size
        if h != self.target_h:
            raise ValueError(f"Unexpected IRS height after resize: {h}")

        if w < self.target_w:
            raise ValueError(f"Unexpected IRS width after resize: {w} < {self.target_w}")

        left = (w - self.target_w) // 2
        return TF.crop(image, top=0, left=left, height=self.target_h, width=self.target_w)


    def __call__(self, left_rgb_image, right_rgb_image, depth_image, normal_image=None):
        left_rgb_image = self.resize(left_rgb_image, InterpolationMode.BILINEAR)
        right_rgb_image = self.resize(right_rgb_image, InterpolationMode.BILINEAR)
        depth_image = self.resize(depth_image, InterpolationMode.NEAREST)

        if normal_image is not None:
            normal_image = self.resize(normal_image, InterpolationMode.BILINEAR)

        left_rgb_image = self.maybe_center_crop_width(left_rgb_image)
        right_rgb_image = self.maybe_center_crop_width(right_rgb_image)
        depth_image = self.maybe_center_crop_width(depth_image)

        if normal_image is not None:
            normal_image = self.maybe_center_crop_width(normal_image)

        left_rgb_tensor = self.to_tensor(left_rgb_image)
        right_rgb_tensor = self.to_tensor(right_rgb_image)
        depth_tensor = self.to_tensor(depth_image)

        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
            return left_rgb_tensor, right_rgb_tensor, depth_tensor, normal_tensor

        return left_rgb_tensor, right_rgb_tensor, depth_tensor

class IRSStereo(Dataset):
    """
    IRS stereo dataset.

    list_file format per line:
        left_img_path right_img_path disp_path [normal_exr_path]

    geometry_mode:
        - "disp": use disparity directly as geometry target (recommended for your current setup)
        - "depth": convert disparity to depth via fx * baseline / disp
    """

    def __init__(
        self,
        data_root='/file_system/goosefsx/public_data/DepthAnything3/qizhe/datasets/IRS/IRS',
        list_file='/file_system/goosefsx/whu/users/hong.li/qizhe/process_datasets/IRS/train.dedup.list',
        transform=True,
        near_plane=1e-5,
        far_plane=80.0,
        fx=725.0087,
        baseline=0.532725,
        geometry_mode="disp",
    ):
        self.data_root = data_root
        self.list_file = list_file
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fx = float(fx)
        self.baseline = float(baseline)
        self.geometry_mode = geometry_mode

        self.samples = []
        self._parse_list(list_file)

        self.transform = SynchronizedTransform_IRS(
            target_h=544,
            target_w=960,
            keep_aspect=False,
            crop_width=False,
        )

    def _parse_list(self, list_file):
        logging.info(f"Loading IRS list from: {list_file}")
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")

        bad = 0
        with open(list_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    bad += 1
                    continue

                left_rel = parts[0]
                right_rel = parts[1]
                disp_rel = parts[2]
                normal_rel = parts[3] if len(parts) >= 4 else None

                left_path = left_rel if os.path.isabs(left_rel) else os.path.join(self.data_root, left_rel)
                right_path = right_rel if os.path.isabs(right_rel) else os.path.join(self.data_root, right_rel)
                disp_path = disp_rel if os.path.isabs(disp_rel) else os.path.join(self.data_root, disp_rel)

                normal_path = None
                if normal_rel is not None and normal_rel.lower() != "none":
                    normal_path = normal_rel if os.path.isabs(normal_rel) else os.path.join(self.data_root, normal_rel)

                self.samples.append(
                    {"left": left_path, "right": right_path, "disp": disp_path, "normal": normal_path}
                )

        if bad > 0:
            logging.warning(f"List parsing done, bad lines skipped: {bad}")

    def _read_rgb_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[..., :3]
        else:
            img = img[..., :3]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    def _smart_read_exr_single_channel(self, path):
        """
        Robust EXR single-channel read.
        If EXR has multiple channels, take max across channels
        (works well when valid signal sits in one channel and others are 0).
        """
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise FileNotFoundError(f"File not found: {path}")

        if len(data.shape) > 2:
            data = np.max(data, axis=2)

        return data.astype(np.float32)

    def _read_disp(self, path):
        disp = self._smart_read_exr_single_channel(path)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        disp = np.abs(disp)
        return disp.astype(np.float32)

    def _read_exr_normal_cam_to_opengl(self, path):
        """
        Read IRS EXR normal (already in camera coordinates), then convert to OpenGL-like convention:
        - +Y up
        - +Z toward camera
        """
        n = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if n is None:
            return None

        if len(n.shape) == 2:
            return None
        if n.shape[2] > 3:
            n = n[..., :3]

        n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB).astype(np.float32)
        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)

        n_min, n_max = float(n.min()), float(n.max())
        if n_min >= -1e-3 and n_max <= 1.0 + 1e-3:
            n = n * 2.0 - 1.0

        norm = np.linalg.norm(n, axis=2, keepdims=True)
        n = n / (norm + 1e-6)

        # CV: y down, z forward(into scene)
        # Want: y up, z toward camera(out of screen)
        n[..., 1] *= -1.0
        n[..., 2] *= -1.0

        norm = np.linalg.norm(n, axis=2, keepdims=True)
        n = n / (norm + 1e-6)

        return n.astype(np.float32)

    def _disp_to_depth(self, disp):
        valid = disp > 1e-6
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = (self.fx * self.baseline) / (disp[valid] + 1e-6)
        return depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        left_rgb = self._read_rgb_img(sample["left"])
        right_rgb = self._read_rgb_img(sample["right"])

        disp = self._read_disp(sample["disp"])

        normal = None
        if sample["normal"] is not None and os.path.exists(sample["normal"]):
            normal = self._read_exr_normal_cam_to_opengl(sample["normal"])

        h_img, w_img = left_rgb.shape[:2]

        if disp.shape[:2] != (h_img, w_img):
            disp = cv2.resize(disp, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        if normal is not None and normal.shape[:2] != (h_img, w_img):
            normal = cv2.resize(normal, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            nrm = np.linalg.norm(normal, axis=2, keepdims=True)
            normal = normal / (nrm + 1e-6)

        # choose geometry source
        if self.geometry_mode == "depth":
            geom = self._disp_to_depth(disp)
        else:
            geom = disp.astype(np.float32)

        left_rgb_image = Image.fromarray(left_rgb)
        right_rgb_image = Image.fromarray(right_rgb)
        geom_image = Image.fromarray(geom.astype(np.float32), mode="F")

        normal_image = None
        if normal is not None:
            normal_u8 = ((np.clip(normal, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
            normal_image = Image.fromarray(normal_u8)

        if self.transform is not None:
            if normal_image is not None:
                left_rgb_tensor, right_rgb_tensor, geom_tensor, normal_tensor = self.transform(
                    left_rgb_image, right_rgb_image, geom_image, normal_image
                )
            else:
                left_rgb_tensor, right_rgb_tensor, geom_tensor = self.transform(
                    left_rgb_image, right_rgb_image, geom_image, None
                )
                normal_tensor = None
        else:
            left_rgb_tensor = transforms.ToTensor()(left_rgb_image)
            right_rgb_tensor = transforms.ToTensor()(right_rgb_image)
            geom_tensor = transforms.ToTensor()(geom_image)
            normal_tensor = transforms.ToTensor()(normal_image) if normal_image is not None else None

        # valid mask
        if self.geometry_mode == "depth":
            valid_depth_mask = (geom_tensor > self.near_plane) & (geom_tensor < self.far_plane)
        else:
            valid_depth_mask = (geom_tensor > 1e-6) & torch.isfinite(geom_tensor)

        left_rgb_tensor = left_rgb_tensor * 2.0 - 1.0
        right_rgb_tensor = right_rgb_tensor * 2.0 - 1.0

        if valid_depth_mask.any():
            flat_geom = geom_tensor[valid_depth_mask].flatten().float()
            min_geom = torch.quantile(flat_geom, 0.02)
            max_geom = torch.quantile(flat_geom, 0.98)

            if min_geom == max_geom:
                depth_tensor = torch.zeros_like(geom_tensor)
                metric_tensor = torch.zeros_like(geom_tensor)
                valid_depth_mask = torch.zeros_like(geom_tensor).bool()
            else:
                geom_tensor = torch.clamp(geom_tensor, min_geom, max_geom)
                geom_tensor[~valid_depth_mask] = max_geom
                metric_tensor = geom_tensor.clone()

                depth_tensor = torch.clamp(
                    (((geom_tensor - min_geom) / (max_geom - min_geom)) * 2.0) - 1.0,
                    -1,
                    1,
                )
        else:
            depth_tensor = torch.zeros_like(geom_tensor)
            metric_tensor = torch.zeros_like(geom_tensor)

        depth_tensor = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze()

        out = {
            "left_rgb": left_rgb_tensor,
            "right_rgb": right_rgb_tensor,
            "depth": depth_tensor,          # normalized geometry (disp or depth)
            "metric": metric_tensor,        # raw geometry (disp or depth)
            "val_mask": valid_depth_mask,
            "domain": "outdoor",
            "left_path": sample["left"],
            "right_path": sample["right"],
            "disp_path": sample["disp"],
            "normal_path": sample["normal"],
        }

        if normal_tensor is not None:
            normal_tensor = normal_tensor * 2.0 - 1.0
            normal_tensor = torch.nn.functional.normalize(normal_tensor, p=2, dim=0)
            normal_tensor[0, ~valid_depth_mask.squeeze()] = 0
            normal_tensor[1, ~valid_depth_mask.squeeze()] = 0
            normal_tensor[2, ~valid_depth_mask.squeeze()] = 0
            out["normals"] = normal_tensor

        return out




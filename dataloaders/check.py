import os
import argparse
from pathlib import Path
from collections import Counter

import torch
from torchvision.utils import save_image


def denorm_rgb(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def depth_vis(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    vis = ((x[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)
    return vis


def normal_vis(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_stats(name, x):
    if not torch.is_tensor(x):
        print(f"{name}: not a tensor, type={type(x)}")
        return

    x_det = x.detach()

    if x_det.numel() == 0:
        print(f"{name}: shape={tuple(x_det.shape)}, dtype={x_det.dtype}, EMPTY tensor")
        return

    x_stat = x_det
    if x_stat.dtype == torch.bool:
        x_stat = x_stat.float()
    elif not torch.is_floating_point(x_stat) and not torch.is_complex(x_stat):
        x_stat = x_stat.float()

    print(
        f"{name}: shape={tuple(x_det.shape)}, dtype={x_det.dtype}, "
        f"min={x_stat.min().item():.6f}, max={x_stat.max().item():.6f}, "
        f"mean={x_stat.mean().item():.6f}"
    )


def check_required_keys(batch, required_keys):
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise KeyError(f"Missing keys in batch: {missing}")


def add_batch_dim_to_sample(sample):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v) and v.dim() >= 1:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def check_batch(batch, batch_name="batch"):
    required_keys = [
        "left_rgb",
        "right_rgb",
        "depth",
        "metric",
        "val_mask",
        "domain",
    ]
    check_required_keys(batch, required_keys)

    print(f"\n===== Checking {batch_name} =====")

    tensor_stats("left_rgb", batch["left_rgb"])
    tensor_stats("right_rgb", batch["right_rgb"])
    tensor_stats("depth", batch["depth"])
    tensor_stats("metric", batch["metric"])
    tensor_stats("val_mask", batch["val_mask"])

    if "normals" in batch:
        tensor_stats("normals", batch["normals"])

    assert batch["left_rgb"].dim() == 4, "left_rgb should be [B,3,H,W]"
    assert batch["right_rgb"].dim() == 4, "right_rgb should be [B,3,H,W]"
    assert batch["depth"].dim() == 4, "depth should be [B,3,H,W]"
    assert batch["metric"].dim() == 4, "metric should be [B,1,H,W] or [B,C,H,W]"
    assert batch["val_mask"].dim() == 4, "val_mask should be [B,1,H,W] or [B,C,H,W]"

    b = batch["left_rgb"].shape[0]
    assert batch["right_rgb"].shape[0] == b
    assert batch["depth"].shape[0] == b
    assert batch["metric"].shape[0] == b
    assert batch["val_mask"].shape[0] == b

    assert batch["left_rgb"].shape[1] == 3
    assert batch["right_rgb"].shape[1] == 3
    assert batch["depth"].shape[1] == 3

    assert batch["left_rgb"].min() >= -1.05 and batch["left_rgb"].max() <= 1.05, "left_rgb out of [-1,1]"
    assert batch["right_rgb"].min() >= -1.05 and batch["right_rgb"].max() <= 1.05, "right_rgb out of [-1,1]"
    assert batch["depth"].min() >= -1.05 and batch["depth"].max() <= 1.05, "depth out of [-1,1]"

    if "normals" in batch:
        assert batch["normals"].min() >= -1.05 and batch["normals"].max() <= 1.05, "normals out of [-1,1]"

    val_ratio = batch["val_mask"].float().mean().item()
    print(f"val_mask ratio: {val_ratio:.6f}")
    if val_ratio <= 0.0:
        print("[WARN] val_mask is empty!")

    domain = batch["domain"]
    if isinstance(domain, (list, tuple)):
        print("domain:", domain[:8], f"(len={len(domain)})")
    else:
        print("domain:", domain)

    for key in [
        "left_rgb_path", "right_rgb_path", "depth_path", "normal_path",
        "left_path", "right_path", "disp_path"
    ]:
        if key in batch:
            val = batch[key]
            if isinstance(val, list):
                print(f"{key}: example -> {val[0]}")
            else:
                print(f"{key}: {val}")

    if "meta" in batch:
        print("meta example:", batch["meta"])

    if "mono_disp" in batch:
        tensor_stats("mono_disp", batch["mono_disp"])


def save_batch_vis(batch, out_dir, prefix, max_items=4):
    os.makedirs(out_dir, exist_ok=True)

    left = denorm_rgb(batch["left_rgb"][:max_items])
    right = denorm_rgb(batch["right_rgb"][:max_items])
    depth = depth_vis(batch["depth"][:max_items]).repeat(1, 3, 1, 1)

    save_image(left, os.path.join(out_dir, f"{prefix}_left_rgb.png"), nrow=2)
    save_image(right, os.path.join(out_dir, f"{prefix}_right_rgb.png"), nrow=2)
    save_image(depth, os.path.join(out_dir, f"{prefix}_depth.png"), nrow=2)

    if "normals" in batch:
        normals = normal_vis(batch["normals"][:max_items])
        save_image(normals, os.path.join(out_dir, f"{prefix}_normals.png"), nrow=2)

    if "val_mask" in batch:
        mask = batch["val_mask"][:max_items].float()
        if mask.shape[1] != 3:
            mask = mask[:, :1].repeat(1, 3, 1, 1)
        save_image(mask, os.path.join(out_dir, f"{prefix}_val_mask.png"), nrow=2)


def inspect_dataset_samples(ds, dataset_name, out_dir, num_samples):
    print(f"\n######## Inspecting {dataset_name} samples ########")
    ds_dir = Path(out_dir) / dataset_name / "samples"
    ds_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(ds))
    for i in range(n):
        sample = add_batch_dim_to_sample(ds[i])
        check_batch(sample, batch_name=f"{dataset_name}_sample_{i}")
        save_batch_vis(sample, str(ds_dir), f"{dataset_name}_sample_{i}")


def inspect_dataset_loader(dl, dataset_name, out_dir, num_batches):
    print(f"\n######## Inspecting {dataset_name} dataloader ########")
    dl_dir = Path(out_dir) / dataset_name / "loader_batches"
    dl_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dl):
        check_batch(batch, batch_name=f"{dataset_name}_loader_batch_{i}")
        save_batch_vis(batch, str(dl_dir), f"{dataset_name}_loader_batch_{i}")

        if i + 1 >= num_batches:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypersim_jsonl", type=str, default="1")
    parser.add_argument("--vkitti_root", type=str, default="1")
    parser.add_argument("--irs_root", type=str, default="1")
    parser.add_argument("--irs_list", type=str, default="1")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_single_samples", type=int, default=10)
    parser.add_argument("--num_loader_batches", type=int, default=10)
    parser.add_argument("--num_mixed_batches", type=int, default=10)

    parser.add_argument("--out_dir", type=str, default="./check_mixed_loader_out_1")

    parser.add_argument("--split_hypersim", type=int, default=9)
    parser.add_argument("--split_vkitti", type=int, default=1)
    parser.add_argument("--split_irs", type=int, default=1)

    parser.add_argument("--check_mixed", action="store_true")

    args = parser.parse_args()

    from load_stereo import (
        HypersimStereoJSONL,
        VirtualKITTI2Stereo,
        IRSStereo,
        MixedDataLoaderN,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    dataset_names = []
    dataloaders = []

    print("Building datasets...")

    if args.hypersim_jsonl:
        ds_h = HypersimStereoJSONL(
            transform=True,
        )
        print(f"Hypersim len: {len(ds_h)}")
        if len(ds_h) == 0:
            raise RuntimeError("Hypersim dataset is empty.")

        dl_h = torch.utils.data.DataLoader(
            ds_h,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        datasets.append(ds_h)
        dataset_names.append("hypersim")
        dataloaders.append(dl_h)

    if args.vkitti_root:
        ds_v = VirtualKITTI2Stereo(

            transform=True,
        )
        print(f"VKITTI len:   {len(ds_v)}")
        if len(ds_v) == 0:
            raise RuntimeError("VKITTI dataset is empty.")

        dl_v = torch.utils.data.DataLoader(
            ds_v,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        datasets.append(ds_v)
        dataset_names.append("vkitti")
        dataloaders.append(dl_v)

    if args.irs_root and args.irs_list:
        ds_i = IRSStereo(

            transform=True,
            geometry_mode="disp",
        )
        print(f"IRS len:      {len(ds_i)}")
        if len(ds_i) == 0:
            raise RuntimeError("IRS dataset is empty.")

        dl_i = torch.utils.data.DataLoader(
            ds_i,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        datasets.append(ds_i)
        dataset_names.append("irs")
        dataloaders.append(dl_i)

    if len(datasets) == 0:
        raise RuntimeError("No dataset enabled. Please provide at least one dataset path.")

    # 1) per-dataset samples
    for name, ds in zip(dataset_names, datasets):
        inspect_dataset_samples(
            ds=ds,
            dataset_name=name,
            out_dir=str(out_dir),
            num_samples=args.num_single_samples,
        )

    # 2) per-dataset dataloader batches
    for name, dl in zip(dataset_names, dataloaders):
        inspect_dataset_loader(
            dl=dl,
            dataset_name=name,
            out_dir=str(out_dir),
            num_batches=args.num_loader_batches,
        )

    # 3) mixed loader
    if args.check_mixed and len(dataloaders) >= 2:
        print("\n######## Inspecting mixed dataloader ########")

        splits = []
        for name in dataset_names:
            if name == "hypersim":
                splits.append(args.split_hypersim)
            elif name == "vkitti":
                splits.append(args.split_vkitti)
            elif name == "irs":
                splits.append(args.split_irs)

        print("Dataset order in mixed loader:", dataset_names)
        print("Splits:", splits)

        mixed_loader = MixedDataLoaderN(dataloaders, splits)

        mixed_dir = Path(out_dir) / "mixed"
        mixed_dir.mkdir(parents=True, exist_ok=True)

        domain_counter = Counter()

        for i, batch in enumerate(mixed_loader):
            d = batch["domain"]
            if isinstance(d, list):
                domain_counter.update(d)
            else:
                domain_counter.update([d])

            check_batch(batch, batch_name=f"mixed_batch_{i}")
            save_batch_vis(batch, str(mixed_dir), f"mixed_batch_{i}")

            if i + 1 >= args.num_mixed_batches:
                break

        print("mixed domain counts:", domain_counter)

    print(f"\nDone. Visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
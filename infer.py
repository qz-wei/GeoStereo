from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision

from GeoStereo.GeoStereo_pipeline import GeoStereoPipeline


def load_image(path: str) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def save_normal(normal: torch.Tensor, path: Path) -> None:
    normal = ((normal.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    torchvision.utils.save_image(normal, str(path))


def save_disparity(disparity: torch.Tensor, path: Path) -> None:
    disparity = disparity.detach().float().cpu()[0, 0].numpy()
    valid = disparity > 1e-3

    if valid.any():
        low, high = np.percentile(disparity[valid], [1, 99])
    else:
        low, high = 0.0, 1.0

    disparity = np.clip((disparity - low) / (high - low + 1e-6), 0.0, 1.0)
    disparity = cv2.applyColorMap(
        (disparity * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    disparity[~valid] = 0
    cv2.imwrite(str(path), disparity)


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoStereo inference")
    parser.add_argument("--model_path", default='./pretrained')
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--save_dir", default="./outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    if device.type == "cpu":
        dtype = torch.float32

    pipeline = GeoStereoPipeline.from_pretrained(
        args.model_path,
        device=device,
        torch_dtype=dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    left = load_image(args.left).to(device)
    right = load_image(args.right).to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.inference_mode():
        output = pipeline(
            image1=left,
            image2=right,
            generator=generator,
            output_type="pt",
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_normal(output.normal, save_dir / "normal.png")
    save_disparity(output.disparity, save_dir / "disparity.png")

    print(f"Normal:    {save_dir / 'normal.png'}")
    print(f"Disparity: {save_dir / 'disparity.png'}")


if __name__ == "__main__":
    main()
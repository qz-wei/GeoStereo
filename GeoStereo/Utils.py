# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, time,torch,torchvision,pickle,trimesh,itertools,datetime,imageio,logging,joblib,importlib,argparse
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import pandas as pd
import open3d as o3d
import cv2
import numpy as np
# from transformations import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)



def set_logging_format(level=logging.INFO):
  importlib.reload(logging)
  FORMAT = '%(message)s'
  logging.basicConfig(level=level, format=FORMAT, datefmt='%m-%d|%H:%M:%S')

set_logging_format()



def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud



def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
  invalid_mask = (depth<zmin)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map



def freeze_model(model):
  model = model.eval()
  for p in model.parameters():
    p.requires_grad = False
  for p in model.buffers():
    p.requires_grad = False
  return model



def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)   #!NOTE KITTI width=1242
  W_resize = round_by_divider(W)
  if H_resize>max_H or W_resize>max_W:
    if H_resize>W_resize:
      W_resize = round_by_divider(W_resize*max_H/H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize*max_W/W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)



def depth_uint8_decoding(depth_uint8, scale=1000):
  depth_uint8 = depth_uint8.astype(float)
  out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
  return out/float(scale)


import torch
import torch.nn.functional as F

def safe_normalize(x, dim=1, eps=1e-8):
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)



def disparity_loss(disp_pred, disp_gt, valid_mask=None, max_disp=192):
    """
    disp_pred/disp_gt: [B,1,H,W]
    valid_mask: bool/float, [B,1,H,W] or [B,H,W] or None
    """
    if valid_mask is None:
        mask = torch.isfinite(disp_gt) & (disp_gt > 0)
    else:
        mask = (valid_mask > 0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask & torch.isfinite(disp_gt) & (disp_gt > 0)

    if max_disp is not None:
        mask = mask & (disp_gt < float(max_disp))

    if mask.sum() < 1:
        # keep graph-connected zero (避免 DDP/AMP 奇怪问题)
        return disp_pred.sum() * 0.0

    return (disp_pred[mask] - disp_gt[mask]).abs().mean()


def compute_disp_loss_full_and_quarter(
    disp_quarter: torch.Tensor | None,   # init_disp_14, typically 1/4
    disp_full: torch.Tensor | None,      # disp_pred, full-res
    disp_gt: torch.Tensor,               # disp_gt_px [B,1,H,W]
    mask_disp: torch.Tensor | None,      # mask bool/float
    max_disp: float = 192.0,
    w_full: float = 1.0,
    w_quarter: float = 0.3,
):
    """
    只计算 disparity loss：full + 1/4（都会对齐到 GT 尺寸再算）
    返回 total_loss（保持与计算图连接）
    """
    assert disp_gt.ndim == 4 and disp_gt.shape[1] == 1, f"disp_gt should be [B,1,H,W], got {disp_gt.shape}"

    # graph-connected zero
    total = disp_gt.sum() * 0.0

    # -------- full-res pred --------
    if disp_full is not None:
        df = disp_full
        if df.ndim == 3:
            df = df.unsqueeze(1)
        if df.shape[-2:] != disp_gt.shape[-2:]:
            scale = float(disp_gt.shape[-1]) / float(df.shape[-1])
            df = F.interpolate(df, size=disp_gt.shape[-2:], mode="bilinear", align_corners=False) * scale
        total = total + w_full * disparity_loss(df, disp_gt, mask_disp, max_disp=max_disp)

    # -------- 1/4 pred --------
    if disp_quarter is not None:
        dq = disp_quarter
        if dq.ndim == 3:
            dq = dq.unsqueeze(1)
        if dq.shape[-2:] != disp_gt.shape[-2:]:
            scale = float(disp_gt.shape[-1]) / float(dq.shape[-1])
            dq = F.interpolate(dq, size=disp_gt.shape[-2:], mode="bilinear", align_corners=False) * scale
        total = total + w_quarter * disparity_loss(dq, disp_gt, mask_disp, max_disp=max_disp)

    return total

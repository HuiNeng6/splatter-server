#!/usr/bin/env python3
"""
COLMAP -> Video Depth Anything -> Open3D voxel reconstruction.

Requirements:
- A COLMAP sparse model folder with: cameras.bin, images.bin, points3D.bin
- Your original images accessible on disk
- Video Depth Anything repo cloned + deps installed + checkpoints downloaded:
    git clone https://github.com/DepthAnything/Video-Depth-Anything
    cd Video-Depth-Anything && pip install -r requirements.txt && bash get_weights.sh

Outputs:
- <colmap_model>_depth/
    depth_vis.mp4         VDA's native grayscale depth visualization
    scales.csv            per-frame alignment stats
    merged_colored.ply    all frames merged, hue-colored by frame index (debug)
    voxels.ply            voxelized combined point cloud
    frames_ply/           per-frame unprojected depth as point clouds (debug)
    _tmp/                 temp files (kept unless --cleanup)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("Missing: opencv-python. Install: pip install opencv-python") from e

try:
    import pycolmap
except ImportError as e:
    raise SystemExit("Missing: pycolmap. Install: pip install pycolmap") from e

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("Missing: open3d. Install: pip install open3d") from e

try:
    import numba
except ImportError as e:
    raise SystemExit("Missing: numba. Install: pip install numba") from e


# ----------------------------
# Utilities
# ----------------------------

def ensure_even_hw(img: np.ndarray) -> np.ndarray:
    """Some codecs want even width/height."""
    h, w = img.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h == h and new_w == w:
        return img
    return img[:new_h, :new_w].copy()


def find_image_paths(colmap_model_dir: Path, images: list, image_root: Optional[Path]) -> List[Path]:
    """Resolve each Image.name to an actual file path."""
    if image_root is not None:
        out = []
        for im in images:
            p = image_root / im.name
            if not p.exists():
                raise FileNotFoundError(f"Image not found under --image_root: {p}")
            out.append(p)
        return out

    parent = colmap_model_dir.parent
    candidates = [
        parent / "images", parent / "Images", parent / "imgs", parent / "rgb",
        parent / "undistorted" / "images", colmap_model_dir / "images",
    ]

    out: List[Path] = []
    for im in images:
        found = None
        direct = parent / im.name
        if direct.exists():
            found = direct
        else:
            for root in candidates:
                p = root / im.name
                if p.exists():
                    found = p
                    break
        if found is None:
            raise FileNotFoundError(f"Could not locate {im.name}")
        out.append(found)
    return out


def write_video_from_frames(frame_paths: List[Path], out_mp4: Path, fps: int) -> Tuple[int, int]:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read image: {frame_paths[0]}")
    first = ensure_even_hw(first)
    h, w = first.shape[:2]

    if out_mp4.exists():
        print("[IO] Input video already exists, skip writing again.")
        return w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at: {out_mp4}")

    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = ensure_even_hw(img)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            print("WARNING! Resized frame {} from {} to {}. Might lose precision.".format(p, img.shape[:2], (w, h)))
        vw.write(img)

    vw.release()
    return w, h


def bilinear_sample(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample depth at subpixel locations."""
    h, w = depth.shape[:2]
    x, y = xy[:, 0], xy[:, 1]
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full((xy.shape[0],), np.nan, dtype=np.float32)
    if not np.any(valid):
        return out

    xv, yv = x[valid], y[valid]
    x0v, y0v, x1v, y1v = x0[valid], y0[valid], x1[valid], y1[valid]

    Ia, Ib = depth[y0v, x0v], depth[y1v, x0v]
    Ic, Id = depth[y0v, x1v], depth[y1v, x1v]

    wa = (x1v - xv) * (y1v - yv)
    wb = (x1v - xv) * (yv - y0v)
    wc = (xv - x0v) * (y1v - yv)
    wd = (xv - x0v) * (yv - y0v)

    out[valid] = (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(np.float32)
    return out


def fit_scale_only(depth_pred: np.ndarray, depth_gt: np.ndarray) -> float:
    """Robust scale-only: median(depth_gt / depth_pred)."""
    eps = 1e-6
    valid = np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > eps) & (depth_gt > eps)
    if np.count_nonzero(valid) < 20:
        return float("nan")
    ratios = depth_gt[valid] / depth_pred[valid]
    if ratios.size < 20:
        return float("nan")
    
    mask = (ratios > np.quantile(ratios, 0.1)) & (ratios < np.quantile(ratios, 0.9))
    ratios = ratios[mask]
    mean_ratio = np.mean(ratios)
    errs = np.abs(ratios - mean_ratio)
    # Re-fit only with best 80%
    keep = errs <= np.quantile(errs, 0.8)
    ratios = ratios[keep]
    mean_ratio = np.mean(ratios)
    return float(mean_ratio)


def fit_affine_trimmed(depth_pred: np.ndarray, depth_gt: np.ndarray, trim_quantile: float = 0.2) -> Tuple[float, float]:
    """Affine fit (a,b) for depth_gt ≈ a*depth_pred + b."""
    eps = 1e-6
    valid = np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred > eps) & (depth_gt > eps)
    if np.count_nonzero(valid) < 50:
        return float("nan"), float("nan")

    x = depth_pred[valid].astype(np.float64)
    y = depth_gt[valid].astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    r = y - (a * x + b)
    abs_r = np.abs(r)
    thr = np.quantile(abs_r, 1.0 - trim_quantile)
    keep = abs_r <= thr
    if np.count_nonzero(keep) < 50:
        return float(a), float(b)

    x2, y2 = x[keep], y[keep]
    A2 = np.stack([x2, np.ones_like(x2)], axis=1)
    a2, b2 = np.linalg.lstsq(A2, y2, rcond=None)[0]
    return float(a2), float(b2)


# ----------------------------
# VDA runner
# ----------------------------

def run_vda_inference(
    vda_repo: Path,
    input_video: Path,
    out_dir: Path,
    encoder: str,
    metric: bool,
    input_size: int,
    max_res: int,
    fp32: bool,
) -> None:
    """Call VDA's run.py - outputs npz + grayscale visualization video."""
    run_py = vda_repo / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Could not find {run_py}")

    cmd = [
        sys.executable, str(run_py),
        "--input_video", str(input_video),
        "--output_dir", str(out_dir),
        "--encoder", encoder,
        "--input_size", str(input_size),
        #"--max_res", str(max_res),
        "--save_npz",
        "--grayscale",
    ]
    if metric:
        cmd.append("--metric")
    if fp32:
        cmd.append("--fp32")

    print("[VDA] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(vda_repo))
    print("[VDA] Inference DONE")


def load_vda_depths(vda_out_dir: Path, num_frames: int) -> List[np.ndarray]:
    """Load depth predictions from VDA's stacked npz output."""
    npzs = sorted([Path(p) for p in glob.glob(str(vda_out_dir / "**" / "*.npz"), recursive=True)])
    if not npzs:
        raise FileNotFoundError(f"No .npz depth outputs found under: {vda_out_dir}")

    if len(npzs) == 1:
        npz_path = npzs[0]
        print(f"[VDA] Loading stacked depths from: {npz_path}")
        d = np.load(str(npz_path))

        arr = None
        for key in ["depth", "pred", "prediction", "arr_0"]:
            if key in d.files:
                arr = d[key]
                break
        if arr is None:
            arr = d[d.files[0]] if len(d.files) == 1 else None
        if arr is None:
            raise RuntimeError(f"Unknown npz structure in {npz_path}. Keys={d.files}")

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if arr.ndim == 3:
            if arr.shape[0] != num_frames:
                raise RuntimeError(f"Depth count mismatch: expected {num_frames}, got {arr.shape[0]}")
            return [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            if num_frames != 1:
                raise RuntimeError(f"Expected {num_frames} frames but got single 2D array")
            return [arr]
        else:
            raise RuntimeError(f"Unexpected depth shape {arr.shape}")

    if len(npzs) != num_frames:
        raise RuntimeError(f"Depth count mismatch: frames={num_frames} vs npz={len(npzs)}")

    depths = []
    for npz_path in npzs:
        d = np.load(str(npz_path))
        arr = None
        for key in ["depth", "pred", "prediction", "arr_0"]:
            if key in d.files:
                arr = d[key]
                break
        if arr is None:
            arr = d[d.files[0]] if len(d.files) == 1 else None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        depths.append(arr)
    
    return depths


def find_vda_vis_video(vda_out_dir: Path) -> Optional[Path]:
    """Find VDA's native visualization video."""
    for pattern in ["*_vis.mp4", "*vis*.mp4", "*.mp4"]:
        matches = list(vda_out_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ----------------------------
# Open3D point cloud / voxel helpers
# ----------------------------

@numba.jit(nopython=True, parallel=True)
def compute_depth_normals(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    radius: int = 3,
) -> np.ndarray:
    """
    Compute surface normals from depth map using multi-scale central differences.
    Samples multiple points within a radius and averages the resulting normals.
    Returns (H, W, 3) normal map in camera space (pointing towards camera).
    """
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    
    for row in numba.prange(h):
        for col in range(w):
            # Skip border pixels
            if row < radius or row >= h - radius or col < radius or col >= w - radius:
                normals[row, col, 2] = -1.0  # Default normal pointing at camera
                continue
            
            d = depth[row, col]
            if d <= 0:
                normals[row, col, 2] = -1.0
                continue
            
            # Accumulate normal estimates from multiple sample distances
            nx_sum = 0.0
            ny_sum = 0.0
            nz_sum = 0.0
            valid_samples = 0
            
            # Sample at multiple distances (1, 2, ..., radius)
            for r in range(1, radius + 1):
                # Get depths at cardinal directions
                d_r = depth[row, col + r]
                d_l = depth[row, col - r]
                d_d = depth[row + r, col]
                d_u = depth[row - r, col]
                
                # Skip if any neighbor has invalid depth
                if d_r <= 0 or d_l <= 0 or d_d <= 0 or d_u <= 0:
                    continue
                
                # Unproject to camera space
                x_r = (col + r - cx) * d_r / fx
                y_r = (row - cy) * d_r / fy
                z_r = d_r
                
                x_l = (col - r - cx) * d_l / fx
                y_l = (row - cy) * d_l / fy
                z_l = d_l
                
                x_d = (col - cx) * d_d / fx
                y_d = (row + r - cy) * d_d / fy
                z_d = d_d
                
                x_u = (col - cx) * d_u / fx
                y_u = (row - r - cy) * d_u / fy
                z_u = d_u
                
                # Central difference tangent vectors (normalized by distance)
                inv_2r = 0.5 / r
                tu_x = (x_r - x_l) * inv_2r
                tu_y = (y_r - y_l) * inv_2r
                tu_z = (z_r - z_l) * inv_2r
                
                tv_x = (x_d - x_u) * inv_2r
                tv_y = (y_d - y_u) * inv_2r
                tv_z = (z_d - z_u) * inv_2r
                
                # Cross product: normal = tv x tu
                nx = tv_y * tu_z - tv_z * tu_y
                ny = tv_z * tu_x - tv_x * tu_z
                nz = tv_x * tu_y - tv_y * tu_x
                
                # Normalize this sample's normal
                norm = np.sqrt(nx * nx + ny * ny + nz * nz)
                if norm < 1e-8:
                    continue
                
                nx /= norm
                ny /= norm
                nz /= norm
                
                # Flip if pointing away from camera
                if nz > 0:
                    nx = -nx
                    ny = -ny
                    nz = -nz
                
                nx_sum += nx
                ny_sum += ny
                nz_sum += nz
                valid_samples += 1
            
            if valid_samples == 0:
                normals[row, col, 2] = -1.0
                continue
            
            # Average and normalize
            nx = nx_sum / valid_samples
            ny = ny_sum / valid_samples
            nz = nz_sum / valid_samples
            
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            if norm < 1e-8:
                normals[row, col, 2] = -1.0
                continue
            
            normals[row, col, 0] = nx / norm
            normals[row, col, 1] = ny / norm
            normals[row, col, 2] = nz / norm
    
    return normals


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    cam_to_world: pycolmap.Rigid3d,
    max_depth: float = 100.0,
    stride: int = 1,
    image_rgb: np.ndarray = None,
    normals_cam: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unproject depth map to world-space 3D points.
    Returns (pts, colors, normals) - each (N, 3) float32 arrays.
    """
    h, w = depth.shape
    #print("[depth_to_pointcloud] depth.shape = ", depth.shape)
    #print("[depth_to_pointcloud] fx = ", fx, "fy = ", fy, "cx = ", cx, "cy = ", cy)
    #print("[depth_to_pointcloud] cam_to_world = ", cam_to_world)

    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.flatten(), vv.flatten()

    z = depth[vv, uu]

    #print("[depth_to_pointcloud] z.shape = ", z.shape, ", mean = ", np.mean(z), ", std = ", np.std(z), ", min = ", np.min(z), ", max = ", np.max(z))
    valid = np.isfinite(z) & (z > 1e-3) & (z < max_depth)
    uu, vv, z = uu[valid], vv[valid], z[valid]

    # Camera coords
    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy

    pts_cam = np.stack([x_cam, y_cam, z], axis=1)  # (N, 3)

    pts_world = cam_to_world * pts_cam
    
    if image_rgb is not None:
        colors = image_rgb[vv, uu]
        colors = colors.astype(np.float32) / 255.
    else:
        colors = np.zeros((pts_world.shape[0], 3), dtype=np.float32)
    
    # Transform normals to world space
    if normals_cam is not None:
        normals_sampled = normals_cam[vv, uu]  # (N, 3) in camera space
        normals_sampled /= np.linalg.norm(normals_sampled, axis=1, keepdims=True)

        # Extract rotation from cam_to_world (normals only need rotation, not translation)
        # pycolmap Rigid3d rotation transforms directions
        normals_world = cam_to_world.rotation * normals_sampled
        normals_world = normals_world.astype(np.float32)

        # Skip points where normal points too sharply sideways, which happens on large depth jumps
        mask = normals_sampled[:, 2] < -0.1
        count_before = pts_world.shape[0]
        pts_world = pts_world[mask]
        colors = colors[mask]
        normals_world = normals_world[mask]
        print(f"Skip {count_before - pts_world.shape[0]} depthmap 3D points where normal points too much to the side")
    else:
        normals_world = np.zeros((pts_world.shape[0], 3), dtype=np.float32)
    
    return pts_world.astype(np.float32), colors, normals_world


def hue_to_rgb(hue: float) -> Tuple[float, float, float]:
    """Convert hue [0,1] to RGB [0,1] with full saturation/value."""
    import colorsys
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


def save_pointcloud_ply(
    pts: np.ndarray,
    path: Path,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> None:
    """Save (N,3) points as PLY, optionally with colors and normals."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def texture_mesh_from_images(
    mesh: o3d.geometry.TriangleMesh,
    images: List,  # pycolmap images
    frame_paths: List[Path],
    aligned_depths: List[np.ndarray],
    max_depth: float = 10.0,
    num_iterations: int = 0,  # 0 = just project colors, no optimization
) -> o3d.geometry.TriangleMesh:
    """
    Texture a mesh by projecting images onto it using Open3D's color map optimization.

    Hard-fails if:
      - Loaded image dims != camera dims
      - Camera model is not pinhole (SIMPLE_PINHOLE or PINHOLE)
    """
    if len(images) == 0:
        print("[TEXTURE][WARN] No images provided, skipping texture projection")
        return mesh

    if not (len(images) == len(frame_paths) == len(aligned_depths)):
        msg = (f"[TEXTURE][ERROR] Length mismatch: "
               f"images={len(images)}, frame_paths={len(frame_paths)}, aligned_depths={len(aligned_depths)}")
        print(msg)
        raise ValueError(msg)

    # Sample a subset of images for efficiency (use every Nth frame)
    max_images = 50
    if len(images) > max_images:
        step = max(1, len(images) // max_images)
        indices = list(range(0, len(images), step))[:max_images]
    else:
        indices = list(range(len(images)))

    print(f"[TEXTURE] Using {len(indices)} images for texture projection...")

    rgbd_images = []
    camera_trajectory = o3d.camera.PinholeCameraTrajectory()

    for i in indices:
        im = images[i]
        img_path = frame_paths[i]
        depth_aligned = aligned_depths[i]

        # --- Camera model check (hard fail) ---
        cam = im.camera
        model_name = getattr(cam.model, "name", str(cam.model))
        if model_name not in ("SIMPLE_PINHOLE", "PINHOLE"):
            msg = (f"[TEXTURE][ERROR] Camera model '{model_name}' is not pinhole for frame index {i}. "
                   f"Only SIMPLE_PINHOLE and PINHOLE are supported (hard fail).")
            print(msg)
            raise ValueError(msg)

        # Load color image
        color_img_bgr = cv2.imread(str(img_path))
        if color_img_bgr is None:
            msg = f"[TEXTURE][ERROR] Could not load image at {img_path} (frame index {i})"
            print(msg)
            raise FileNotFoundError(msg)

        color_img = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)

        # --- Dimension checks (hard fail) ---
        img_h, img_w = color_img.shape[:2]
        cam_w, cam_h = int(cam.width), int(cam.height)

        if (img_w != cam_w) or (img_h != cam_h):
            msg = (f"[TEXTURE][ERROR] Image dimensions do not match camera dimensions for index {i}:\n"
                   f"  image:  {img_w}x{img_h}  path={img_path}\n"
                   f"  camera: {cam_w}x{cam_h}  model={model_name}\n"
                   f"Hard failing as requested (no automatic scaling).")
            print(msg)
            raise ValueError(msg)

        if depth_aligned.ndim != 2:
            msg = (f"[TEXTURE][ERROR] Depth map for index {i} must be HxW (2D). "
                   f"Got shape={depth_aligned.shape}.")
            print(msg)
            raise ValueError(msg)

        if (depth_aligned.shape[0] != img_h) or (depth_aligned.shape[1] != img_w):
            msg = (f"[TEXTURE][ERROR] Depth dimensions do not match image dimensions for index {i}:\n"
                   f"  depth: {depth_aligned.shape[1]}x{depth_aligned.shape[0]}\n"
                   f"  image: {img_w}x{img_h}\n"
                   f"Hard failing as requested (no resizing).")
            print(msg)
            raise ValueError(msg)

        # Create Open3D images
        color_o3d = o3d.geometry.Image(color_img.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_aligned.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,          # assumes depth is already in meters
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        rgbd_images.append(rgbd)

        # Get camera intrinsics (pinhole only)
        if model_name == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            fx, fy = f, f
        elif model_name == "PINHOLE":
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        else:
            # unreachable due to earlier check
            raise RuntimeError("Unexpected camera model after validation")

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cam_w,
            height=cam_h,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )

        # Get camera extrinsics (world -> camera)
        cam_from_world = im.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsic
        cam_params.extrinsic = extrinsic
        camera_trajectory.parameters.append(cam_params)

    if len(rgbd_images) == 0:
        print("[TEXTURE][WARN] No valid images loaded, skipping texture projection")
        return mesh

    if not mesh.has_vertex_normals():
        print("[TEXTURE][WARN] Mesh had no vertex normals; computing them for visibility checks...")
        mesh.compute_vertex_normals()

    print(f"[TEXTURE] Running color map optimization (iterations={num_iterations})...")

    opt = o3d.pipelines.color_map.NonRigidOptimizerOption(
        maximum_iteration=100,#int(num_iterations),
        depth_threshold_for_visibility_check=0.2,
        depth_threshold_for_discontinuity_check=0.5,
        maximum_allowable_depth=float(max_depth),
    )

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory, opt
        )

    print("[TEXTURE] Texture projection complete")
    return mesh



def create_mesh_from_pointcloud(
    pts: np.ndarray,
    normals: np.ndarray,
    colors: Optional[np.ndarray] = None,
    depth: int = 8,
) -> o3d.geometry.TriangleMesh:
    """Create mesh from point cloud with normals using Poisson reconstruction."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    
    # Remove low-density vertices (cleaning up mesh boundaries)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Remove vertices not near the voxel points
    max_distance = 0.2
    tree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices)
    points = np.asarray(pcd.points)
    vertices_to_remove = np.zeros(len(verts), dtype=bool)
    for i, vert in enumerate(verts):
        [k, idx, _] = tree.search_knn_vector_3d(vert, 1)
        if k > 0:
            dist = np.linalg.norm(vert - points[idx[0]])
            if dist > max_distance:
                vertices_to_remove[i] = True
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def voxelize_points(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelize points and return voxel centers."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxels = voxel_grid.get_voxels()
    centers = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])
    return centers.astype(np.float32)


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """
    Convert a normal vector to a quaternion that rotates Z-axis to align with normal.
    Returns quaternion as (w, x, y, z).
    """
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0])
    
    dot = np.dot(z_axis, normal)
    
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif dot < -0.9999:
        # 180 degree rotation around X axis
        return np.array([0.0, 1.0, 0.0, 0.0])
    
    # Rotation axis = cross(z, normal)
    axis = np.cross(z_axis, normal)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Rotation angle
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Quaternion from axis-angle
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def save_gaussian_splat_ply(
    path: Path,
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    disc_scale: float = 0.5,
    init_opacity: float = 0.9,
    thickness_ratio: float = 0.15,
) -> None:
    """
    Save voxel data as Gaussian splat PLY file.
    
    Each Gaussian is a flat disc oriented by the normal.
    Scale is the Gaussian sigma, so visible extent is ~3*sigma.
    - Disc sigma: voxel_size * disc_scale (3-sigma ≈ voxel_size)
    - Thickness sigma: voxel_size * thickness_ratio (very flat)
    
    PLY format compatible with 3D Gaussian Splatting tools.
    """
    n_points = positions.shape[0]
    
    # Compute scales (in log space as expected by most splat implementations)
    # Scale is Gaussian sigma; visible extent is roughly 3*sigma
    disc_sigma = voxel_size * disc_scale
    thickness_sigma = disc_sigma * thickness_ratio
    
    # Scales stored as log(scale) in the PLY
    scale_disc = np.log(disc_sigma)
    scale_thin = np.log(thickness_sigma)
    #scale_thin = scale_disc
    
    # Compute quaternions from normals (disc lies perpendicular to normal)
    quaternions = np.zeros((n_points, 4), dtype=np.float32)
    for i, normal in enumerate(normals):
        quaternions[i] = normal_to_quaternion(normal)
        #quaternions[i] = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Convert colors from [0,1] to spherical harmonics DC term
    # SH DC coefficient = (color - 0.5) / C0 where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0
    
    # Build PLY data
    # Standard 3DGS PLY format properties
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    # Add 45 f_rest coefficients (for SH degree 3, set to 0)
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    data = np.zeros(n_points, dtype=dtype)
    
    # Position
    data['x'] = positions[:, 0]
    data['y'] = positions[:, 1]
    data['z'] = positions[:, 2]
    
    # Normal
    data['nx'] = normals[:, 0]
    data['ny'] = normals[:, 1]
    data['nz'] = normals[:, 2]
    
    # SH DC color
    data['f_dc_0'] = sh_dc[:, 0]
    data['f_dc_1'] = sh_dc[:, 1]
    data['f_dc_2'] = sh_dc[:, 2]
    
    # f_rest coefficients (all zeros for basic color)
    for i in range(45):
        data[f'f_rest_{i}'] = 0.0
    
    # Opacity (inverse sigmoid of 0.9 for high opacity)
    # sigmoid(x) = 0.9 => x = log(0.9/0.1) ≈ 2.197
    data['opacity'] = np.log(init_opacity / (1.0 - init_opacity))
    
    # Scales: disc plane (x,y) larger, z thin
    data['scale_0'] = scale_disc
    data['scale_1'] = scale_disc
    data['scale_2'] = scale_thin
    
    # Rotation quaternion (w, x, y, z)
    data['rot_0'] = quaternions[:, 0]
    data['rot_1'] = quaternions[:, 1]
    data['rot_2'] = quaternions[:, 2]
    data['rot_3'] = quaternions[:, 3]
    
    # Write PLY file
    with open(path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
        for i in range(45):
            header += f"property float f_rest_{i}\n"
        header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

class GridCell:
    point_sum: np.ndarray
    point_count: int

    normal_sum: np.ndarray
    normal_count: int

    color_sum: np.ndarray
    color_count: int

    confidence: float

    def __init__(self):
        self.point_sum = np.array([0.0, 0.0, 0.0])
        self.point_count = 0
        self.normal_sum = np.array([0.0, 0.0, 0.0])
        self.normal_count = 0
        self.color_sum = np.array([0.0, 0.0, 0.0])
        self.color_count = 0
        self.confidence = 0.0
    
    def add_point(self, point: np.ndarray, normal: np.ndarray = None, color: np.ndarray = None):
        self.point_sum += point
        self.point_count += 1
        if normal is not None:
            self.normal_sum += normal
            self.normal_count += 1
        if color is not None:
            self.color_sum += color
            self.color_count += 1
    
    def add_confidence(self, value: float):
        self.confidence += value
    
    def get_point_mean(self) -> np.ndarray:
        return self.point_sum / self.point_count if self.point_count > 0 else None
    
    def get_normal_mean(self) -> np.ndarray:
        return self.normal_sum / self.normal_count if self.normal_count > 0 else None
    
    def get_color_mean(self) -> np.ndarray:
        return self.color_sum / self.color_count if self.color_count > 0 else None
    
    def get_means(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_point_mean(), self.get_normal_mean(), self.get_color_mean()


@numba.jit(nopython=True, parallel=True)
def compute_voxel_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute voxel indices for all points. Returns (N, 3) int32 array."""
    n = len(points)
    result = np.empty((n, 3), dtype=np.int32)
    for i in numba.prange(n):
        result[i, 0] = int(np.round(points[i, 0] / voxel_size))
        result[i, 1] = int(np.round(points[i, 1] / voxel_size))
        result[i, 2] = int(np.round(points[i, 2] / voxel_size))
    return result


@numba.jit(nopython=True)
def compute_empty_voxel_indices(
    camera_pos: np.ndarray,
    ray_dirs_norm: np.ndarray,
    ray_lengths: np.ndarray,
    valid_rays: np.ndarray,
    voxel_size: float,
    max_depth: float
) -> np.ndarray:
    """Compute voxel indices for empty space along rays before depth surfaces.
    
    Returns array of shape (M, 3) containing integer voxel indices.
    """
    max_steps = int(max_depth / voxel_size)
    n_rays = len(ray_lengths)
    
    # Pre-allocate worst case: all rays * all steps
    max_entries = n_rays * max_steps
    result = np.empty((max_entries, 3), dtype=np.int32)
    count = 0
    
    for step in range(1, max_steps + 1):
        t = step * voxel_size
        margin = voxel_size * 0.5
        
        any_valid = False
        for i in range(n_rays):
            if valid_rays[i] and t < ray_lengths[i] - margin:
                any_valid = True
                # Compute empty point position
                px = camera_pos[0] + t * ray_dirs_norm[i, 0]
                py = camera_pos[1] + t * ray_dirs_norm[i, 1]
                pz = camera_pos[2] + t * ray_dirs_norm[i, 2]
                # Compute voxel index
                result[count, 0] = int(np.round(px / voxel_size))
                result[count, 1] = int(np.round(py / voxel_size))
                result[count, 2] = int(np.round(pz / voxel_size))
                count += 1
        
        if not any_valid:
            break
    
    return result[:count]


class VoxelGrid:
    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
        self.cells = dict()  # spatial hash: (ix, iy, iz) -> GridCell

    def _voxel_index(self, point: np.ndarray) -> tuple:
        # Compute voxel grid index by rounding to nearest voxel centroid
        idx = np.round(point / self.voxel_size).astype(int)
        return (idx[0], idx[1], idx[2])

    def add_point(self, point: np.ndarray, normal: np.ndarray = None, color: np.ndarray = None, confidence: float = 1.0):
        voxel_idx = self._voxel_index(point)
        if voxel_idx not in self.cells:
            self.cells[voxel_idx] = GridCell()
        self.cells[voxel_idx].add_point(point, normal, color)
        self.cells[voxel_idx].add_confidence(confidence)

    def add_empty(self, point: np.ndarray, subtractive_value: float):
        """Mark a voxel as potentially empty by adding negative confidence."""
        voxel_idx = self._voxel_index(point)
        if voxel_idx not in self.cells:
            self.cells[voxel_idx] = GridCell()
        self.cells[voxel_idx].add_confidence(subtractive_value)

    def add_empty_batch(self, voxel_indices: np.ndarray, subtractive_value: float):
        """Batch mark voxels as empty from pre-computed indices (M, 3) array."""
        for i in range(len(voxel_indices)):
            voxel_idx = (voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2])
            if voxel_idx not in self.cells:
                self.cells[voxel_idx] = GridCell()
            self.cells[voxel_idx].add_confidence(subtractive_value)

    def add_points_batch(self, points: np.ndarray, normals: np.ndarray = None, colors: np.ndarray = None, confidence: float = 1.0):
        """Batch add points with pre-computed voxel indices for efficiency."""
        voxel_indices = compute_voxel_indices(points, self.voxel_size)
        cells = self.cells
        
        for i in range(len(points)):
            voxel_idx = (int(voxel_indices[i, 0]), int(voxel_indices[i, 1]), int(voxel_indices[i, 2]))
            if voxel_idx not in cells:
                cells[voxel_idx] = GridCell()
            cell = cells[voxel_idx]
            cell.point_sum[0] += points[i, 0]
            cell.point_sum[1] += points[i, 1]
            cell.point_sum[2] += points[i, 2]
            cell.point_count += 1
            cell.confidence += confidence
        
        if normals is not None:
            for i in range(len(normals)):
                voxel_idx = (int(voxel_indices[i, 0]), int(voxel_indices[i, 1]), int(voxel_indices[i, 2]))
                cell = cells[voxel_idx]
                cell.normal_sum[0] += normals[i, 0]
                cell.normal_sum[1] += normals[i, 1]
                cell.normal_sum[2] += normals[i, 2]
                cell.normal_count += 1
        
        if colors is not None:
            for i in range(len(colors)):
                voxel_idx = (int(voxel_indices[i, 0]), int(voxel_indices[i, 1]), int(voxel_indices[i, 2]))
                cell = cells[voxel_idx]
                cell.color_sum[0] += colors[i, 0]
                cell.color_sum[1] += colors[i, 1]
                cell.color_sum[2] += colors[i, 2]
                cell.color_count += 1

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("colmap_model_dir", type=str, help="Folder with cameras.bin/images.bin/points3D.bin")
    ap.add_argument("--image_root", type=str, default=None,
                    help="Root directory to resolve COLMAP image names")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output folder. Default: <colmap_model_dir>_depth")
    ap.add_argument("--vda_repo", type=str, default=os.environ.get("VDA_REPO", ""),
                    help="Path to cloned Video-Depth-Anything repo (or set env VDA_REPO)")
    ap.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--metric_model", action="store_true",
                    help="Use VDA metric model weights")
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--max_res", type=int, default=1280)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--fit_shift", action="store_true",
                    help="Fit affine (scale+shift) instead of scale-only")
    ap.add_argument("--voxel_size", type=float, default=0.1,
                    help="Voxel size in meters for the output voxel grid")
    ap.add_argument("--voxel_min_points", type=int, default=3,
                    help="Discard voxels with fewer points than this threshold")
    ap.add_argument("--voxel_min_confidence", type=float, default=3.0,
                    help="Discard voxels with confidence below this threshold")
    ap.add_argument("--subtractive_scanning", action="store_true",
                    help="Enable subtractive scanning: rays mark voxels before depth as empty")
    ap.add_argument("--subtractive_value", type=float, default=-0.1,
                    help="Certainty contribution for empty voxels (must be negative)")
    ap.add_argument("--max_depth", type=float, default=10.0,
                    help="Max depth in meters for point cloud generation")
    ap.add_argument("--stride", type=int, default=10,
                    help="Pixel stride for unprojection (higher = sparser, faster)")
    ap.add_argument("--cleanup", action="store_true", help="Delete temp folder after success")
    ap.add_argument("--reuse_vda", action="store_true", help="Reuse VDA output if it already exists")
    ap.add_argument("--save_voxel_mesh", action="store_true",
                    help="Generate mesh from voxel points using Poisson reconstruction")
    ap.add_argument("--texture_mesh", action="store_true",
                    help="Texture the mesh by projecting images onto it (requires --save_voxel_mesh)")
    ap.add_argument("--texture_iterations", type=int, default=0,
                    help="Color map optimization iterations (0 = simple projection, >0 = non-rigid optimization)")
    ap.add_argument("--save_splat", action="store_true",
                    help="Save Gaussian splat PLY with flat disc Gaussians per voxel")
    args = ap.parse_args()

    if args.subtractive_value >= 0:
        raise ValueError("--subtractive_value must be negative")

    colmap_dir = Path(args.colmap_model_dir).resolve()
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP model directory not found: {colmap_dir}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else Path(str(colmap_dir) + "_depth")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_ply_dir = out_dir / "frames_ply"
    tmp_dir = out_dir / "_tmp"
    frames_ply_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    vda_repo = Path(args.vda_repo).resolve() if args.vda_repo else None
    if vda_repo is None or not vda_repo.exists():
        raise FileNotFoundError("Provide --vda_repo or set env VDA_REPO")

    print("[COLMAP] Reading model...")
    reconstruction = pycolmap.Reconstruction(colmap_dir)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3d = reconstruction.points3D

    img_list = sorted(images.values(), key=lambda im: im.name)

    image_root = Path(args.image_root).resolve() if args.image_root else None
    frame_paths = find_image_paths(colmap_dir, img_list, image_root=image_root)
    print(f"[COLMAP] Found {len(frame_paths)} frames.")

    # Build temp input MP4 for VDA
    input_video = tmp_dir / "input.mp4"
    print(f"[IO] Writing temp video: {input_video}")
    video_w, video_h = write_video_from_frames(frame_paths, input_video, fps=10)

    # Run VDA
    vda_out = tmp_dir / "vda_out"

    if vda_out.exists() and args.reuse_vda:
        print("[VDA] VDA output already exists, skip running again.")
    else:
        if vda_out.exists():
            shutil.rmtree(vda_out)
        vda_out.mkdir(parents=True, exist_ok=True)

        run_vda_inference(
            vda_repo=vda_repo,
            input_video=input_video,
            out_dir=vda_out,
            encoder=args.encoder,
            metric=args.metric_model,
            input_size=args.input_size,
            max_res=args.max_res,
            fp32=args.fp32,
        )
    
    # Copy VDA's visualization video to output
    vda_vis = find_vda_vis_video(vda_out)
    if vda_vis:
        out_vis = out_dir / "depth_vis.mp4"
        shutil.copy2(vda_vis, out_vis)
        print(f"[VDA] Copied visualization: {out_vis}")

    # Load VDA depth predictions
    depth_preds = load_vda_depths(vda_out, num_frames=len(img_list))
    print(f"[VDA] Loaded {len(depth_preds)} depth frames.")


    # CSV for alignment stats
    scales_csv = out_dir / "scales.csv"
    csv_f = scales_csv.open("w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["frame_idx", "image_name", "num_anchors", "scale_a", "shift_b", "mode"])

    grid = VoxelGrid(voxel_size=args.voxel_size)
    num_frames = len(img_list)
    aligned_depths = []  # Store aligned depths for texture projection

    print("[ALIGN] Aligning depths and unprojecting to point clouds...")
    for idx, (im, img_path, depth_pred_raw) in enumerate(zip(img_list, frame_paths, depth_preds)):
        if idx % 2 > 0:
            continue

        _time_start = time.time()
        depth_pred = depth_pred_raw
        # depth must be exactly same dimensions as the image
        img_w, img_h = im.camera.width, im.camera.height
        if depth_pred.shape[0] != img_h or depth_pred.shape[1] != img_w:
        #    #print("Resizing depth map: depth_pred.shape = ", depth_pred.shape, "img_w = ", img_w, "img_h = ", img_h)
            depth_pred = cv2.resize(depth_pred, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        cam = cameras[im.camera_id]
        cam_from_world = im.cam_from_world()
        cam_to_world = cam_from_world.inverse()

        # Get camera intrinsics (PINHOLE: fx, fy, cx, cy)
        params = np.array(cam.params)
        if cam.model.name in ("PINHOLE",):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            # Fallback: assume first param is focal, center is image center
            fx = fy = params[0]
            cx, cy = w / 2, h / 2

        # Build anchor correspondences from COLMAP tracks for alignment
        xyzs, xys_keep, track_lengths = [], [], []
        for pt2d in im.points2D:
            pid = pt2d.point3D_id
            if pid < 0 or pid not in points3d:
                continue
            pt3d = points3d[pid]
            xyzs.append(np.array(pt3d.xyz))
            xys_keep.append(np.array(pt2d.xy))

        if xyzs:
            Xw = np.stack(xyzs, axis=0).astype(np.float64)
            xy = np.stack(xys_keep, axis=0).astype(np.float64)
            Xc = cam_from_world * Xw
            depth_gt = Xc[:, 2].astype(np.float32)
            #depth_samp = bilinear_sample(depth_pred, xy.astype(np.float32))
            #nearest instaed:
            depth_samp = depth_pred[xy[:, 1].astype(np.int32), xy[:, 0].astype(np.int32)]
            ok = np.isfinite(depth_gt) & np.isfinite(depth_samp) & (depth_gt > 1e-3) & (depth_samp > 1e-6)
            depth_gt_ok = depth_gt[ok]
            depth_samp_ok = depth_samp[ok]
        else:
            depth_gt_ok = np.array([], dtype=np.float32)
            depth_samp_ok = np.array([], dtype=np.float32)

        mode = "scale"
        a, b = float("nan"), 0.0

        if depth_gt_ok.size < 50:
            print(f"Frame {idx} has less than 50 valid depth points, skipping")
            continue

        if depth_gt_ok.size >= 50:
            if args.fit_shift and depth_gt_ok.size >= 50:
                mode = "affine"
                a, b = fit_affine_trimmed(depth_samp_ok, depth_gt_ok)
                if not np.isfinite(a):
                    mode = "scale"
                    a = fit_scale_only(depth_samp_ok, depth_gt_ok)
                    b = 0.0
            else:
                a = fit_scale_only(depth_samp_ok, depth_gt_ok)
                b = 0.0
            #print("[ALIGN] mode = ", mode, "a = ", a, "b = ", b)
        
        #a = 1.0
        #b = 0.0

        if not np.isfinite(a):
            print(f"Frame {idx} has no valid depth points, skipping")
            a, b, mode = 1.0, 0.0, "none"

        depth_aligned = (a * depth_pred + b).astype(np.float32)
        #print("[ALIGN] depth_aligned.shape = ", depth_aligned.shape, "mean = ", np.mean(depth_aligned), "std = ", np.std(depth_aligned), "min = ", np.min(depth_aligned), "max = ", np.max(depth_aligned))
        writer.writerow([idx, im.name, int(depth_gt_ok.size), float(a), float(b), mode])
        aligned_depths.append(depth_aligned if mode != "none" else None)

        if mode == "none":
            continue

        image_rgb = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        if image_rgb.shape[0] != img_h or image_rgb.shape[1] != img_w:
            image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            print("WARNING! Resized image {} from {} to {}. Might lose precision.".format(img_path, image_rgb.shape[:2], (img_w, img_h)))
        
        # Compute normals from depth map
        _time = time.time()
        normals_cam = compute_depth_normals(depth_aligned, fx, fy, cx, cy)
        #print(f"[TIME] compute_depth_normals took {time.time() - _time} seconds")

        # Unproject to world-space point cloud
        _time = time.time()
        pts, colors, normals = depth_to_pointcloud(
            depth_aligned, fx, fy, cx, cy, cam_to_world,
            max_depth=args.max_depth, stride=args.stride,
            image_rgb=image_rgb, normals_cam=normals_cam
        )
        #print(f"[TIME] depth_to_pointcloud took {time.time() - _time} seconds")

        if pts.size > 0:
            # Assign hue-based color for this frame (rainbow gradient across sequence)
            #hue = idx / max(1, num_frames - 1)
            #rgb = hue_to_rgb(hue)
            #colors = np.tile(np.array(rgb, dtype=np.float32), (pts.shape[0], 1))
            
            # Save per-frame PLY for debugging
            #_time = time.time()
            #frame_ply = frames_ply_dir / f"frame_{idx:06d}.ply"
            #save_pointcloud_ply(pts, frame_ply, colors=colors, normals=normals)
            #print(f"[TIME] save_pointcloud_ply took {time.time() - _time} seconds")

            _time = time.time()
            grid.add_points_batch(pts, normals=normals, colors=colors)
            #print(f"[TIME] grid.add_points_batch took {time.time() - _time} seconds")

            # Subtractive scanning: mark voxels along rays before depth as empty
            if args.subtractive_scanning:
                _time = time.time()
                camera_pos = (cam_to_world * np.array([[0.0, 0.0, 0.0]]))[0]
                ray_dirs = pts - camera_pos
                ray_lengths = np.linalg.norm(ray_dirs, axis=1)
                valid_rays = ray_lengths > args.voxel_size
                ray_dirs_norm = np.zeros_like(ray_dirs)
                ray_dirs_norm[valid_rays] = ray_dirs[valid_rays] / ray_lengths[valid_rays, np.newaxis]

                empty_voxel_indices = compute_empty_voxel_indices(
                    camera_pos, ray_dirs_norm, ray_lengths, valid_rays,
                    args.voxel_size, args.max_depth
                )
                grid.add_empty_batch(empty_voxel_indices, args.subtractive_value)
                #print(f"[TIME] subtractive_scanning took {time.time() - _time} seconds ({len(empty_voxel_indices)} empty voxels)")
        
        #print(f"[TIME] total iteration took {time.time() - _time_start} seconds")

        if (idx + 1) % 10 == 0 or (idx + 1) == len(img_list):
            print(f"  frame {idx+1}/{len(img_list)}, points: {pts.shape[0] if pts.size else 0}")

    csv_f.close()

    if grid.cells:

        voxel_points = []
        voxel_normals = []
        voxel_colors = []
        for voxel_idx, cell in grid.cells.items():
            if cell.point_count < args.voxel_min_points:
                continue
            if cell.confidence < args.voxel_min_confidence:
                continue
            point_mean, normal_mean, color_mean = cell.get_means()
            if point_mean is None:
                continue

            voxel_points.append(point_mean)

            if normal_mean is not None:
                voxel_normals.append(normal_mean)
            if color_mean is not None:
                voxel_colors.append(color_mean)

        if voxel_points:
            voxel_points = np.array(voxel_points)
            voxel_normals = np.array(voxel_normals) if voxel_normals else None
            voxel_colors = np.array(voxel_colors) if voxel_colors else None
            
            # Normalize the averaged normals
            if voxel_normals is not None:
                norms = np.linalg.norm(voxel_normals, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                voxel_normals = voxel_normals / norms
            
            voxels_ply = out_dir / "grid_voxels.ply"
            save_pointcloud_ply(voxel_points, voxels_ply, colors=voxel_colors, normals=voxel_normals)
            print(f"  Saved: {voxels_ply}")
            
            # Generate mesh if requested
            if args.save_voxel_mesh and voxel_normals is not None:
                print("[MESH] Generating mesh from voxel points...")
                mesh = create_mesh_from_pointcloud(voxel_points, voxel_normals, colors=voxel_colors)
                
                # Texture the mesh by projecting images onto it
                if args.texture_mesh:
                    # Filter out frames where alignment failed (None depths)
                    valid_indices = [i for i, d in enumerate(aligned_depths) if d is not None]
                    valid_images = [img_list[i] for i in valid_indices]
                    valid_paths = [frame_paths[i] for i in valid_indices]
                    valid_depths = [aligned_depths[i] for i in valid_indices]
                    
                    mesh = texture_mesh_from_images(
                        mesh, valid_images, valid_paths, valid_depths,
                        max_depth=args.max_depth,
                        num_iterations=args.texture_iterations
                    )
                
                mesh_ply = out_dir / "grid_mesh.ply"
                o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_vertex_normals=True)
                print(f"  Saved: {mesh_ply}")
            
            # Generate Gaussian splat PLY if requested
            if args.save_splat and voxel_normals is not None and voxel_colors is not None:
                print("[SPLAT] Generating Gaussian splat PLY...")
                splat_ply = out_dir / "grid_splat.ply"
                save_gaussian_splat_ply(
                    splat_ply, voxel_points, voxel_normals, voxel_colors,
                    voxel_size=args.voxel_size
                )
                print(f"  Saved: {splat_ply}")

    print(f"\n[DONE] Output: {out_dir}")
    print(f"       - depth_vis.mp4         (VDA visualization)")
    print(f"       - scales.csv            (alignment stats)")
    print(f"       - merged_colored.ply    (all frames, hue-colored by frame)")
    print(f"       - grid_voxels.ply       (voxelized reconstruction, averaged points)")
    print(f"       - grid_mesh.ply         (mesh from voxel points, if --save_voxel_mesh, textured if --texture_mesh)")
    print(f"       - grid_splat.ply        (Gaussian splat PLY for 3DGS, if --save_splat)")
    print(f"       - frames_ply/           (per-frame point clouds)")

    if args.cleanup:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("[CLEANUP] Removed temp folder.")


if __name__ == "__main__":
    main()

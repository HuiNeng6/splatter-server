#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pycolmap


@dataclass
class CenterCropArgs:
    model: str | Path
    images: str | Path
    out: str | Path

    # Reproject triangulated observations from points3D (keeps a consistent COLMAP model after crops).
    update_points2D: bool = True

    # Undistortion options (PyCOLMAP mirrors COLMAP’s undistort behavior) 
    blank_pixels: float = 0.0
    min_scale: float = 0.2
    max_scale: float = 2.0
    max_image_size: int = -1
    roi_min_x: float = 0.0
    roi_min_y: float = 0.0
    roi_max_x: float = 1.0
    roi_max_y: float = 1.0

    # If True, abort when cropping-only cannot make cx/cy exactly centered (integer crop limitation).
    # If False, we keep the true cx/cy after crop and warn.
    require_exact_center: bool = False

    # Numpy float tolerance for the exact-center check.
    center_tolerance: float = 1e-6


def make_center_crop(w: int, h: int, cx: float, cy: float) -> Tuple[int, int, int, int]:
    """
    Choose the largest crop fully inside the image that is symmetric around (cx, cy).
    After cropping with integer (left, top), the new principal becomes:
      cx' = cx - left
      cy' = cy - top
    and the new image size is (new_w, new_h).
    """
    half_w = min(cx, w - cx)
    half_h = min(cy, h - cy)

    new_w = int(np.floor(2.0 * half_w))
    new_h = int(np.floor(2.0 * half_h))

    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    left = int(np.round(cx - new_w / 2.0))
    top = int(np.round(cy - new_h / 2.0))

    left = max(0, min(left, w - new_w))
    top = max(0, min(top, h - new_h))

    return left, top, new_w, new_h


def set_camera_principal_point(cam: pycolmap.Camera, cx_new: float, cy_new: float) -> None:
    params = np.array(cam.params, dtype=float).copy()
    pp = cam.principal_point_idxs()
    if len(pp) != 2:
        raise ValueError(f"Camera model {cam.model} doesn't expose exactly 2 principal point indices.")
    params[pp[0]] = cx_new
    params[pp[1]] = cy_new
    cam.params = params


def to_simple_pinhole_or_pinhole(cam: pycolmap.Camera, camera_id: int) -> pycolmap.Camera:
    """
    Ensure camera model is SIMPLE_PINHOLE or PINHOLE (no distortion parameters).
    Uses focal/principal point accessors. 
    """
    fx = float(cam.focal_length_x)
    fy = float(cam.focal_length_y)
    cx = float(cam.principal_point_x)
    cy = float(cam.principal_point_y)

    if np.isfinite(fx) and np.isfinite(fy) and abs(fx - fy) < 1e-9:
        model = pycolmap.CameraModelId.SIMPLE_PINHOLE
        params = np.array([0.5 * (fx + fy), cx, cy], dtype=float)
    else:
        model = pycolmap.CameraModelId.PINHOLE
        params = np.array([fx, fy, cx, cy], dtype=float)

    return pycolmap.Camera(
        camera_id=camera_id,
        model=model,
        width=int(cam.width),
        height=int(cam.height),
        params=params,
    )


def reproject_triangulated_points(recon: pycolmap.Reconstruction) -> None:
    """
    Update only points2D that observe a 3D point by reprojecting the 3D point
    into the (possibly changed) camera. Keeps poses unchanged. 
    """
    for image_id, image in recon.images.items():
        if not image.has_pose:
            continue

        cam = recon.cameras[image.camera_id]
        T = image.cam_from_world().matrix()  # 3x4 
        R = T[:, :3]
        t = T[:, 3]

        obs_idxs = image.get_observation_point2D_idxs()
        for idx in obs_idxs:
            p2d = image.point2D(idx)
            pid = int(p2d.point3D_id)
            if pid not in recon.points3D:
                continue

            X = np.array(recon.points3D[pid].xyz, dtype=float)
            Xc = R @ X + t

            uv = cam.img_from_cam(Xc.reshape(3, 1))
            if uv is None:
                continue

            p2d.xy = np.array([float(uv[0, 0]), float(uv[1, 0])], dtype=float)


def main(args: CenterCropArgs) -> None:
    """
    Undistort all images using pycolmap and then crop to make cx,cy == w/2,h/2 (cropping-only).
    Writes:
      <out>/images/...
      <out>/sparse/{cameras,images,points3D}.{bin|txt}
    """
    model_dir = Path(args.model)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    out_img_dir = out_dir / "images"
    out_model_dir = out_dir / "sparse"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction 
    recon = pycolmap.Reconstruction(str(model_dir))

    undist_opts = pycolmap.UndistortCameraOptions(
        blank_pixels=float(args.blank_pixels),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        max_image_size=int(args.max_image_size),
        roi_min_x=float(args.roi_min_x),
        roi_min_y=float(args.roi_min_y),
        roi_max_x=float(args.roi_max_x),
        roi_max_y=float(args.roi_max_y),
    )

    # Cache crop per camera_id for consistency (same camera intrinsics => same crop)
    crop_by_cam: Dict[int, Tuple[int, int, int, int]] = {}

    for image_id, image in recon.images.items():
        in_path = images_dir / image.name
        if not in_path.exists():
            raise FileNotFoundError(f"Missing image: {in_path}")

        distorted_cam = recon.cameras[image.camera_id]

        bmp = pycolmap.Bitmap.read(str(in_path), as_rgb=True)  # 
        if bmp is None:
            raise RuntimeError(f"Failed to read image: {in_path}")

        und_bmp, und_cam = pycolmap.undistort_image(undist_opts, bmp, distorted_cam)  # 

        w = int(und_cam.width)
        h = int(und_cam.height)
        cx = float(und_cam.principal_point_x)  # 
        cy = float(und_cam.principal_point_y)

        if image.camera_id not in crop_by_cam:
            crop_by_cam[image.camera_id] = make_center_crop(w, h, cx, cy)

        left, top, new_w, new_h = crop_by_cam[image.camera_id]

        # Crop the bitmap via numpy roundtrip
        arr = und_bmp.to_array()  #   shape (H,W,3)
        cropped = arr[top:top + new_h, left:left + new_w]
        out_bmp = pycolmap.Bitmap.from_array(np.ascontiguousarray(cropped))  # 

        # Update camera intrinsics for the crop
        # New principal point after cropping:
        cx_crop = cx - left
        cy_crop = cy - top

        # Check whether cropping-only gives exact center
        rx = (new_w / 2.0) - cx_crop
        ry = (new_h / 2.0) - cy_crop
        if abs(rx) > args.center_tolerance or abs(ry) > args.center_tolerance:
            msg = (
                f"cam {image.camera_id}: cropping-only cannot perfectly center principal point. "
                f"Residual to exact center: (rx,ry)=({rx:.6f},{ry:.6f}) px. "
                f"Consider adding a subpixel translation (resample) if you require exact centering."
            )
            if args.require_exact_center:
                raise RuntimeError(msg)
            else:
                print("[WARN]", msg)

        und_cam.width = int(new_w)
        und_cam.height = int(new_h)
        set_camera_principal_point(und_cam, cx_crop, cy_crop)

        # Convert to SIMPLE_PINHOLE/PINHOLE (no distortion) after undistortion 
        recon.cameras[image.camera_id] = to_simple_pinhole_or_pinhole(und_cam, image.camera_id)

        # Write output image
        out_path = out_img_dir / image.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = out_bmp.write(str(out_path))
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

    if args.update_points2D:
        reproject_triangulated_points(recon)

    # Write updated model 
    recon.write(str(out_model_dir))
    print(f"Done.\nImages: {out_img_dir}\nModel:  {out_model_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Undistort + crop to center principal point (PyCOLMAP).")
    ap.add_argument("--model", required=True, help="Input COLMAP model dir (e.g. sparse/0)")
    ap.add_argument("--images", required=True, help="Input images directory")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--no_update_points2D", action="store_true",
                    help="Do NOT reproject triangulated points2D from points3D.")

    ap.add_argument("--blank_pixels", type=float, default=0.0)
    ap.add_argument("--min_scale", type=float, default=0.2)
    ap.add_argument("--max_scale", type=float, default=2.0)
    ap.add_argument("--max_image_size", type=int, default=-1)
    ap.add_argument("--roi_min_x", type=float, default=0.0)
    ap.add_argument("--roi_min_y", type=float, default=0.0)
    ap.add_argument("--roi_max_x", type=float, default=1.0)
    ap.add_argument("--roi_max_y", type=float, default=1.0)

    ap.add_argument("--require_exact_center", action="store_true",
                    help="Fail if cropping-only cannot make cx/cy exactly centered.")
    ap.add_argument("--center_tolerance", type=float, default=1e-6)

    ns = ap.parse_args()

    main(CenterCropArgs(
        model=ns.model,
        images=ns.images,
        out=ns.out,
        update_points2D=not ns.no_update_points2D,
        blank_pixels=ns.blank_pixels,
        min_scale=ns.min_scale,
        max_scale=ns.max_scale,
        max_image_size=ns.max_image_size,
        roi_min_x=ns.roi_min_x,
        roi_min_y=ns.roi_min_y,
        roi_max_x=ns.roi_max_x,
        roi_max_y=ns.roi_max_y,
        require_exact_center=ns.require_exact_center,
        center_tolerance=ns.center_tolerance,
    ))

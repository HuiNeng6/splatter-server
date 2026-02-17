from pathlib import Path
import pycolmap
import preprocessing
import filter_splats
import convert_splat
import argparse
import time
import sys
import shutil
import os
from subprocess import run
import open3d as o3d
from preprocessing import run_bundle_adjustment, set_colmap_points_from_pointcloud, cleanup_rec_cameras

LICHTFELD_BIN = os.environ.get("LICHTFELD_BIN", "/app/LichtFeld-Studio/build/LichtFeld-Studio")
LICHTFELD_CONFIG = os.environ.get("LICHTFELD_CONFIG", "config/lichtfeld_optimization_params.json")
LICHTFELD_CONFIG_VDA = os.environ.get("LICHTFELD_CONFIG_VDA", "config/lichtfeld_optimization_params_vda.json")
VDA_REPO = os.environ.get("VDA_REPO", "/app/Video-Depth-Anything")

def run_vda_depth(colmap_dir: Path, image_root: Path, output_dir: Path, voxel_size: float = 0.1) -> Path:
    """Run colmap_vda_depth.py to generate initial splat from depth estimation."""
    script_path = Path(__file__).parent / "colmap_vda_depth.py"
    cmd = [
        sys.executable, str(script_path),
        str(colmap_dir),
        "--image_root", str(image_root),
        "--output_dir", str(output_dir),
        "--vda_repo", VDA_REPO,
        "--encoder", "vitl",
        "--metric",
        "--max_depth", "2.5",
        "--stride", "10",
        "--voxel_size", str(voxel_size),
        "--subtractive_scanning",
        "--subtractive_value", "-0.5",
        #"--fit_shift",
        "--save_splat",
        "--save_voxel_mesh",
        #"--texture_mesh",
        "--reuse_vda",
        "--fp32",
    ]

    print(f"[VDA] Running: {' '.join(cmd)}")
    run(cmd, check=True)
    return output_dir / "grid_splat.ply"


def train_splat(colmap_dir: Path, output_dir: Path, images_dir: Path = None,
                iterations: int = 20000,
                enable_sparsity: bool = False, sparsify_steps: int = 15000,
                init_ply: Path = None):
    if not images_dir:
        images_dir = colmap_dir / "images"

    cmd = [
        LICHTFELD_BIN,
        "-d", str(colmap_dir),
        "-o", str(output_dir),
        "-r", "2",
        "-i", str(iterations),
        "--images", os.path.abspath(str(images_dir)),
        "--config", LICHTFELD_CONFIG_VDA if init_ply is not None else LICHTFELD_CONFIG,
        "--headless",
        #"--train"
    ]

    if enable_sparsity:
        cmd.append("--enable-sparsity")
        cmd.append(f"--sparsify-steps")
        cmd.append(str(sparsify_steps))

    if init_ply is not None:
        cmd.extend(["--init", str(init_ply)])
    print(f"Running command: {' '.join(cmd)}")
    run(cmd)

def scan_main(job_root: Path, scan_id: str, iterations: int = 20000,
              enable_sparsity: bool = False, sparsify_steps: int = 15000,
              reuse_trained: bool = False, use_vda: bool = False,
              convert_splat: bool = False, convert_sog: bool = False,
              bundle_adjust: bool = True) -> None:
    start_time = time.time()

    total_iters = iterations
    if enable_sparsity:
        total_iters += sparsify_steps
    
    splat_filename = f"splat_{total_iters}"
    splat_dir = job_root / "refined" / "local" / scan_id / "splat"
    if reuse_trained and (splat_dir / f"{splat_filename}.ply").exists():
        print("Splat reuse enabled, skip ahead to filtering")
    else:
        if not use_vda:
            preprocessing.main(job_root, scan_id, bundle_adjust=bundle_adjust)
            init_ply = None
        else:
            # Don't run the full preprocessing since the colmap 3D point
            # outlier filtering clears 2D/3D point relations.
            # These are needed by the VDA densification to scale depth maps to fit point cloud.

            sfm_dir = job_root / "refined" / "local" / scan_id / "sfm"
            frames_dir = job_root / "datasets" / scan_id / "Frames"
            
            vda_output_dir = job_root / "refined" / "local" / scan_id / "vda_depth"
            init_ply = run_vda_depth(sfm_dir, frames_dir, vda_output_dir, voxel_size=0.1)
            #init_ply = None

            processed_dir = job_root / "refined" / "local" / scan_id / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            rec = pycolmap.Reconstruction()
            rec.read(str(sfm_dir))
            rec = cleanup_rec_cameras(rec, frames_dir=frames_dir)
            if bundle_adjust:
                rec = run_bundle_adjustment(rec)
            
            vda_pointcloud = vda_output_dir / "grid_voxels.ply"
            pcd = o3d.io.read_point_cloud(vda_pointcloud)
            print(f"[VDA] override colmap with {len(pcd.points)} points from VDA voxel grid")
            set_colmap_points_from_pointcloud(rec, pcd)
            rec.write(processed_dir)
            print(f"[VDA] Using init splat: {init_ply}")

            
            dense_dir = job_root / "refined" / "local" / scan_id / "dense"
            shutil.rmtree(dense_dir, ignore_errors=True)
            dense_dir.mkdir(parents=True, exist_ok=True)
            pycolmap.undistort_images(
                output_path=str(dense_dir),
                input_path=str(processed_dir),
                image_path=str(frames_dir)
            )
        
        if bundle_adjust:
            images_dir = job_root / "refined" / "local" / scan_id / "dense" / "images"
            colmap_dir = job_root / "refined" / "local" / scan_id / "dense" / "sparse"
        else:
            #colmap_dir = job_root / "refined" / "local" / scan_id / "sfm"
            colmap_dir = job_root / "refined" / "local" / scan_id / "processed"
            images_dir = job_root / "datasets" / scan_id / "Frames"
        
        train_splat(colmap_dir, splat_dir, images_dir, iterations=iterations,
                    enable_sparsity=enable_sparsity, sparsify_steps=sparsify_steps, init_ply=init_ply)

    filtered_ply = splat_dir / f"{splat_filename}.filtered.ply"
    filter_args = argparse.Namespace(
        input_ply=splat_dir / f"{splat_filename}.ply",
        output_ply=filtered_ply,
        colmap_path=colmap_dir,
        overwrite=True,
        min_opacity=-2.5,
        max_size=0.8,
        min_size=0.002,
        min_view_count=5,
        view_min_depth=0.1,
        view_max_depth=3.0,
        outlier_radius=None,
        #outlier_radius=0.3,
        #outlier_neighbors=30
    )

    filter_splats.main(filter_args)

    if not filtered_ply.exists():
        raise FileNotFoundError(f"Filtered PLY was not produced: {filtered_ply}")

    if convert_splat:
        convert_splat.convert_ply_to_splat(filtered_ply)
    if convert_sog:
        convert_splat.convert_ply_to_sog(filtered_ply)

    end_time = time.time()
    print(f"Local splatting DONE in {end_time - start_time} seconds: {scan_id}")

def main(job_root: Path,
         scan_ids: list[str],
         iterations: int = 20000,
         enable_sparsity: bool = False,
         sparsify_steps: int = 15000,
         reuse_trained: bool = False,
         convert_splat: bool = False,
         convert_sog: bool = False,
         use_vda: bool = False
) -> None:
    print(f"Use vda: {use_vda}")
    for scan_id in scan_ids:
        scan_main(job_root, scan_id, iterations=iterations,
                  enable_sparsity=enable_sparsity, sparsify_steps=sparsify_steps,
                  reuse_trained=reuse_trained, use_vda=use_vda)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--scan-id", type=str, default=None)
    parser.add_argument("--all-scan-ids", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--reuse-trained", action="store_true", default=False)
    parser.add_argument("--enable-sparsity", action="store_true", default=False)
    parser.add_argument("--sparsify-steps", type=int, default=15000)
    parser.add_argument("--convert-splat", action="store_true", default=False,
                        help="Also output as .splat format")
    parser.add_argument("--convert-sog", action="store_true", default=False,
                        help="Also output as .sog format")
    parser.add_argument("--vda", action="store_true", default=False,
                        help="Use Video-Depth-Anything to generate initial splat for training")

    args = parser.parse_args()
    print(f"args: {args}")

    if not args.scan_id and not args.all_scan_ids:
        parser.error("Either --scan-id or --all-scan-ids is required")

    if args.all_scan_ids:
        scan_ids = [f.name for f in (args.job_root / "datasets").iterdir() if f.is_dir()]
    else:
        scan_ids = [args.scan_id]
    main(job_root=args.job_root,
         scan_ids=scan_ids,
         iterations=args.iterations,
         enable_sparsity=args.enable_sparsity,
         sparsify_steps=args.sparsify_steps,
         reuse_trained=args.reuse_trained,
         convert_splat=args.convert_splat,
         convert_sog=args.convert_sog,
         use_vda=args.vda)
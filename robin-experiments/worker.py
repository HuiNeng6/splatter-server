# worker.py
"""
Job processor for splat training pipeline.

Supports two job types:
- local_splat: Train splats for individual scans
- global_splat: Combine, partition, and convert splats

Output file naming (for domain data upload):
- local_splat: local_splat_{scan_id}.local_splat_ply
- global_splat partitions:
    - .splat files -> refined_splat_partition_{suffix}.refined_splat
    - .sog files -> refined_splat_sog_partition_{suffix}.refined_splat_sog
"""

import os
import shutil
from pathlib import Path

from jobs import upload_job_result
import local_main
import global_main
from artifact_naming import rename_for_domain_upload

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"


def process_job(conn, job: dict):
    job_id = job["id"]
    job_type = job["job_type"]
    input_params = job["input"] or {}
    
    print(f"[Worker] Processing {job_type} job {job_id}")
    
    # Job root is the input directory for this job
    job_root = INPUT_DIR / job_id
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if job_type == "local_splat":
            process_local_splat(job_root, output_dir, input_params)
        elif job_type == "global_splat":
            process_global_splat(job_root, output_dir, input_params)
        else:
            raise ValueError(f"Unknown job type: {job_type}")
        
        # Signal completion - server will upload output files
        upload_job_result(conn, job_id)
        print(f"[Worker] Job {job_id} completed successfully")
        
    except Exception as e:
        print(f"[Worker] Job {job_id} failed: {e}")
        raise


def process_local_splat(job_root: Path, output_dir: Path, params: dict):
    """Train splats for individual scans."""
    scan_ids = params.get("scan_ids")
    if not scan_ids:
        # Auto-discover scan IDs from datasets folder
        datasets_dir = job_root / "datasets"
        if datasets_dir.exists():
            scan_ids = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
        else:
            raise ValueError("No scan_ids provided and datasets folder not found")
    
    if not scan_ids:
        raise ValueError("No scans found to process")
    
    iterations = params.get("iterations", 20000)
    enable_sparsity = params.get("enable_sparsity", False)
    sparsify_steps = 15000 if enable_sparsity else 0
    reuse_trained = params.get("reuse_trained", False)
    
    print(f"[local_splat] Processing {len(scan_ids)} scans: {scan_ids}")
    
    local_main.main(
        job_root=job_root,
        scan_ids=scan_ids,
        iterations=iterations,
        enable_sparsity=enable_sparsity,
        sparsify_steps=sparsify_steps,
        reuse_trained=reuse_trained
    )
    
    # Copy output splats to output directory with domain naming convention
    # Output: local_splat_{scan_id}.local_splat_ply
    # Domain: name="local_splat_{scan_id}.local_splat_ply", data_type="local_splat_ply"
    total_iters = iterations + sparsify_steps
    for scan_id in scan_ids:
        splat_dir = job_root / "refined" / "local" / scan_id / "splat"
        splat_file = splat_dir / f"splat_{total_iters}.filtered.ply"
        if splat_file.exists():
            dest = output_dir / f"local_splat_{scan_id}.local_splat_ply"
            shutil.copy2(splat_file, dest)
            print(f"[local_splat] Output: {dest.name}")


def process_global_splat(job_root: Path, output_dir: Path, params: dict):
    """Combine, partition, and convert splats."""
    scan_ids = params.get("scan_ids")
    if not scan_ids:
        # Auto-discover from datasets folder
        datasets_dir = job_root / "datasets"
        if datasets_dir.exists():
            scan_ids = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
        else:
            raise ValueError("No scan_ids provided and datasets folder not found")
    
    if not scan_ids:
        raise ValueError("No scans found to process")
    
    base_filename = params.get("base_filename", "splat_20000")
    sparsify_steps = params.get("sparsify_steps", 0)
    use_filtered = params.get("use_filtered", True)
    do_partition = params.get("partition", True)
    partition_size = params.get("partition_size", 2.0)
    do_convert_splat = params.get("convert_to_splat", True)
    do_convert_sog = params.get("convert_to_sog", False)
    
    print(f"[global_splat] Processing {len(scan_ids)} scans: {scan_ids}")
    
    result = global_main.main(
        job_root=job_root,
        scan_ids=scan_ids,
        base_filename=base_filename,
        sparsify_steps=sparsify_steps,
        use_filtered=use_filtered,
        do_partition=do_partition,
        partition_size=partition_size,
        do_convert_splat=do_convert_splat,
        do_convert_sog=do_convert_sog
    )
    
    # Copy outputs to output directory with domain naming convention
    global_dir = job_root / "refined" / "global"
    if not global_dir.exists():
        print(f"[global_splat] Warning: global output dir not found: {global_dir}")
        return
    
    for f in global_dir.iterdir():
        if not f.is_file():
            continue
        
        # Rename partition files for domain upload
        # Pattern: combined_splat_partition_{size}_{x}_{z}.splat
        # Output: refined_splat_partition_{size}_{x}_{z}.refined_splat
        dest_name = rename_for_domain_upload(f.name)
        if dest_name:
            shutil.copy2(f, output_dir / dest_name)
            print(f"[global_splat] Output: {dest_name}")
    
    print(f"[global_splat] Result: {result}")



"""
Splat training HTTP server - single container solution.

Endpoints:
  POST /job     Submit a job (download -> process -> upload)
  GET  /health  Health check

Request body for /job:
  {
    "processing_type": "local_splat" | "global_splat" | "full",
    "domain_id": "...",
    "data_ids": ["id1", "id2"],
    "access_token": "...",          // or use DOMAIN_ACCESS_TOKEN env var
    "api_url": "...",               // or use DOMAIN_API_URL env var  
    "dds_url": "...",               // or use DOMAIN_DDS_URL env var
    
    // Processing options:
    "iterations": 20000,
    "enable_sparsity": false,
    "reuse_trained": false,
    "use_filtered": true,
    "partition": true,
    "partition_size": 2.0,
    "convert_to_splat": true,
    "convert_to_sog": false,
    
    // Local mode (skip domain download/upload):
    "local_job_root": "/path/to/job"
  }
"""

import json
import os
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

import local_main
import global_main
import combine_splats
from domain_client import DomainClient
from artifact_naming import rename_for_domain_upload

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

_job_running = False


class JobHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/health":
            status = "busy" if _job_running else "ok"
            self._send_json(200, {"status": status})
        else:
            self._send_json(404, {"error": "Not found"})
    
    def do_POST(self):
        if self.path != "/job":
            self._send_json(404, {"error": "Not found"})
            return
        
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            print(f"Body: {body}")
            params = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return
        
        try:
            print("params: ", params)
            result = process_job_request(params)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            print(f"Error: {e}")
            traceback.print_exc()
        except Exception as e:
            self._send_json(500, {"error": str(e)})
            print(f"Error: {e}")
            traceback.print_exc()


def process_job_request(params: dict) -> dict:
    """Process a job request - download, process, upload."""
    global _job_running
    _job_running = True
    try:
        return _process_job(params)
    finally:
        _job_running = False


def mp4_to_frames(mp4_path, frames_path, filename_prefix=""):
    capture = cv2.VideoCapture(mp4_path)
    frame_count = 0
    print("Unpacking mp4 to frames:", mp4_path, "->", frames_path)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        img_path = f"{frames_path}/{filename_prefix}{frame_count:06d}.jpg"
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, frame)
        frame_count += 1
    print(f"Unpacked {frame_count} frames from mp4")
    capture.release()


def _process_job(params: dict) -> dict:
    processing_type = params.get("processing_type", "full")
    if processing_type not in ["local_splat", "global_splat", "full"]:
        raise ValueError(f"Invalid processing_type: {processing_type}. Use 'local_splat', 'global_splat', or 'full'")
    
    # Local mode - skip domain download/upload
    local_job_root = params.get("local_job_root")
    if local_job_root:
        job_root = Path(local_job_root)
        if not job_root.exists():
            raise ValueError(f"local_job_root does not exist: {job_root}")
        output_dir = job_root / "output"
        domain_client = None
        domain_id = None
        job_id = "local"
    else:
        # Domain mode - need credentials
        domain_id = params.get("domain_id")
        data_ids = params.get("data_ids", [])
        
        if not domain_id:
            raise ValueError("Missing 'domain_id'")
        if not data_ids:
            raise ValueError("Missing 'data_ids'")
        
        # Get domain client
        domain_server_url = params.get("domain_server_url") or os.environ.get("DOMAIN_SERVER_URL")
        access_token = params.get("access_token") or os.environ.get("DOMAIN_ACCESS_TOKEN")
        
        if not all([domain_server_url, access_token]):
            raise ValueError("Missing domain credentials. Provide api_url, dds_url, access_token in request or env vars")
        
        domain_client = DomainClient(domain_server_url, access_token)
        
        # Create job directories
        job_id = uuid.uuid4().hex[:12]
        job_root = DATA_DIR / "jobs" / job_id
        output_dir = DATA_DIR / "output" / job_id
        datasets_dir = job_root / "datasets"
        input_dir = job_root / "input"
        local_refined_dir = job_root / "refined" / "local"
        global_refined_dir = job_root / "refined" / "global"
        
        job_root.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        local_refined_dir.mkdir(parents=True, exist_ok=True)
        global_refined_dir.mkdir(parents=True, exist_ok=True)
        
        # Download input data
        print(f"[job:{job_id}] Downloading {len(data_ids)} files from domain {domain_id}...")
        download_count = domain_client.download_domain_data(domain_id, data_ids, input_dir)
        if download_count == 0:
            raise ValueError("No data downloaded from domain")
        
        refinement_id = "None" # Same timestamp as the refinement manifest domain data name suffix
        for file in input_dir.iterdir():
            if file.is_file():
                # Example
                # dmt_recording_2025-12-01_11-37-53_5492dd0f-232d-45d9-9ed9-546a8bbddfe7.dmt_recording_mp4
                data_type = file.name.split('.')[-1]
                filename = file.name.split('.')[0]
                data_id = filename.split('_')[-1]
                base_name = filename.replace(f'_{data_id}', '')
                scan_id = '_'.join(base_name.split('_')[-2:])
                print(f"Downloaded file type: {data_type}, for scan {scan_id}")

                renaming = {
                    "dmt_recording_mp4": datasets_dir / scan_id / "Frames.mp4",
                    "refined_scan_zip": datasets_dir / scan_id / "RefinedScan.zip",
                    "local_splat_ply": local_refined_dir / scan_id / "splat" / "splat.filtered.ply",
                    "refined_manifest_json": global_refined_dir / "refined_manifest.json"
                }
                if data_type not in renaming:
                    print(f"Unknown data type: {data_type}, skip")
                    continue
                renamed_file = renaming[data_type]
                target_dir = renamed_file.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, str(renamed_file))    

                if data_type == "refined_scan_zip":
                    sfm_dir = local_refined_dir / scan_id / "sfm"
                    sfm_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(str(renamed_file), 'r') as zip_ref:
                        zip_ref.extractall(sfm_dir)
                elif data_type == "dmt_recording_mp4":
                    frames_dir = datasets_dir / scan_id / "Frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    mp4_to_frames(str(renamed_file), str(frames_dir), f"{scan_id}_")
                elif data_type == "local_splat_ply":
                    # Create the dataset dir for scan even though we don't put the file here
                    # Ugly, but since _process_job enumerates sub folders to determine scan ids.
                    (datasets_dir / scan_id).mkdir(parents=True, exist_ok=True)
                elif data_type == "refined_manifest_json":
                    # For global manifest json, the refinement ID is in the name, same format as scan ID in other files.
                    # Currently we rely on this coming in correctly from DMT, as the FIRST data ID in the list of dataIDs in the request
                    refinement_id = scan_id
                    

        print(f"[job:{job_id}] Downloaded {download_count} files")
    
    # Discover scan IDs
    if not datasets_dir.exists():
        raise ValueError(f"datasets folder not found: {datasets_dir}")
    
    scan_ids = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
    if not scan_ids:
        raise ValueError("No scans found in datasets folder")
    
    print(f"[job:{job_id}] Found {len(scan_ids)} scans: {scan_ids}")
    
    # Processing parameters
    iterations = params.get("iterations", 20000)
    enable_sparsity = params.get("enable_sparsity", False)
    sparsify_steps = 15000 if enable_sparsity else 0
    reuse_trained = params.get("reuse_trained", False)
    use_filtered = params.get("use_filtered", True)
    do_partition = params.get("partition", True)
    partition_size = params.get("partition_size", 2.0)
    do_convert_splat = params.get("convert_to_splat", True)
    do_convert_sog = params.get("convert_to_sog", True)
    
    result = {"job_id": job_id, "scan_ids": scan_ids}
    
    # Step 1: Local splat training
    if processing_type in ["local_splat", "full"]:
        print(f"[job:{job_id}] Running local_splat...")
        local_main.main(
            job_root=job_root,
            scan_ids=scan_ids,
            iterations=iterations,
            enable_sparsity=enable_sparsity,
            sparsify_steps=sparsify_steps,
            reuse_trained=reuse_trained,
            convert_splat=do_convert_splat,
            convert_sog=do_convert_sog
        )
        
        # Copy local outputs
        total_iters = iterations + sparsify_steps
        for scan_id in scan_ids:
            splat_dir = job_root / "refined" / "local" / scan_id / "splat"
            if do_convert_splat or do_convert_sog:
                splat_file = combine_splats.get_splat_path(job_root, scan_id, use_filtered, f"splat_{total_iters}")
                if splat_file:
                    dest = output_dir / f"local_splat_{scan_id}.local_splat"
                    shutil.copy2(splat_file, dest)
                    print(f"[job:{job_id}] Output: {dest.name}")
                if sog_file:
                    dest = output_dir / f"local_splat_sog_{scan_id}.local_splat_sog"
                    shutil.copy2(sog_file, dest)
                    print(f"[job:{job_id}] Output: {dest.name}")
            else:
                # Only upload PLY if neither sog or splat is on (since file size is much larger)
                ply_file = splat_dir / f"splat_{total_iters}.filtered.ply"
                if ply_file.exists():
                    dest = output_dir / f"local_splat_ply_{scan_id}.local_splat_ply"
                    shutil.copy2(ply_file, dest)
                    print(f"[job:{job_id}] Output: {dest.name}")
    
    # Step 2: Global splat processing
    if processing_type in ["global_splat", "full"]:
        print(f"[job:{job_id}] Running global_splat...")
        global_result = global_main.main(
            job_root=job_root,
            scan_ids=scan_ids,
            base_filename=f"splat_{iterations + sparsify_steps}" if processing_type == "full" else "splat",
            sparsify_steps=sparsify_steps,
            use_filtered=use_filtered,
            do_partition=do_partition,
            partition_size=partition_size,
            do_convert_splat=do_convert_splat,
            do_convert_sog=do_convert_sog
        )
        result.update(global_result)
        
        # Copy global outputs with domain naming
        global_dir = job_root / "refined" / "global"
        if global_dir.exists():
            for f in global_dir.iterdir():
                if not f.is_file():
                    continue
                dest_name = rename_for_domain_upload(f.name)
                if dest_name:
                    base_name = '.'.join(dest_name.split('.')[:-1])
                    ext = dest_name.split('.')[-1]
                    dest_name = f"{base_name}_{refinement_id}.{ext}"
                    shutil.copy2(f, output_dir / dest_name)
                    print(f"[job:{job_id}] Output: {dest_name}")
    
    # Step 3: Upload to domain
    if domain_client and domain_id:
        print(f"[job:{job_id}] Uploading outputs to domain {domain_id}...")
        uploaded = domain_client.upload_domain_data(domain_id, output_dir)
        result["uploaded_count"] = len(uploaded)
        result["uploaded_ids"] = [d.get("id") for d in uploaded]
        print(f"[job:{job_id}] Uploaded {len(uploaded)} files")
    
    result["success"] = True
    print(f"[job:{job_id}] Done!")
    return result


def run_server(host: str = "0.0.0.0", port: int = 8080):
    server = ThreadingHTTPServer((host, port), JobHandler)
    print(f"[server] Listening on http://{host}:{port}")
    print(f"[server] POST /job to submit jobs, GET /health for health check")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.host, args.port)

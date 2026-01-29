"""
Domain data client for downloading/uploading data from/to the domain server.

This is a simplified Python port of the Rust posemesh-domain-http client.
"""

import os
import uuid
import requests
from pathlib import Path
from typing import Optional


def escape_quotes(s: str) -> str:
    """Escape quotes in a string for Content-Disposition header."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_multipart_body(parts: list[dict]) -> tuple[bytes, str]:
    """
    Build a multipart/form-data body with custom Content-Disposition fields.
    
    Each part dict should have:
        - name: field name
        - data_type: custom data-type field
        - domain_id: custom domain-id field
        - id: optional, custom id field (for updates)
        - data: file-like object or bytes
    
    Returns:
        Tuple of (body_bytes, content_type_header)
    """
    boundary = f"----PythonFormBoundary{uuid.uuid4().hex}"
    
    body_parts = []
    for part in parts:
        # Build Content-Disposition with custom fields matching Go implementation
        disposition = f'form-data; name="{escape_quotes(part["name"])}"'
        disposition += f'; data-type="{escape_quotes(part["data_type"])}"'
        if part.get("id"):
            disposition += f'; id="{escape_quotes(part["id"])}"'
        disposition += f'; domain-id="{escape_quotes(part["domain_id"])}"'
        
        part_header = f'--{boundary}\r\n'
        part_header += f'Content-Type: application/octet-stream\r\n'
        part_header += f'Content-Disposition: {disposition}\r\n'
        part_header += '\r\n'
        
        data = part["data"]
        if hasattr(data, 'read'):
            data = data.read()
        
        body_parts.append(part_header.encode('utf-8'))
        body_parts.append(data)
        body_parts.append(b'\r\n')
    
    body_parts.append(f'--{boundary}--\r\n'.encode('utf-8'))
    
    body = b''.join(body_parts)
    content_type = f'multipart/form-data; boundary={boundary}'
    
    return body, content_type


class DomainClient:
    """Client for interacting with the domain data server."""
    
    def __init__(self, domain_server_url: str, access_token: str):
        """
        Initialize domain client.
        
        Args:
            domain_server_url: Base URL for domain data server
            access_token: Bearer token for authentication
        """
        self.domain_server_url = domain_server_url.rstrip('/')
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {access_token}",
        })
    
    def download_domain_data(self, domain_id: str, data_ids: list[str], output_dir: Path) -> int:
        """
        Download domain data files by IDs.
        
        Args:
            domain_id: Domain ID to download from
            data_ids: List of data IDs to download
            output_dir: Directory to save downloaded files
        
        Returns:
            Number of files downloaded
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for data_id in data_ids:
            try:
                # Get metadata
                print(f"[domain] Getting metadata for {data_id}")
                meta_url = f"{self.domain_server_url}/api/v1/domains/{domain_id}/data/{data_id}"
                meta_resp = self.session.get(meta_url)
                meta_resp.raise_for_status()
                meta = meta_resp.json()
                print(f"[domain] Metadata: {meta}")
                
                # Download actual data with streaming
                download_url = f"{self.domain_server_url}/api/v1/domains/{domain_id}/data/{data_id}?raw=true"
                
                filename = f"{meta['name']}_{data_id}.{meta['data_type']}"
                filepath = output_dir / filename
                
                prev_pct = 0
                with self.session.get(download_url, stream=True) as data_resp:
                    data_resp.raise_for_status()
                    total_size = int(data_resp.headers.get('content-length', 0))
                    if total_size == 0 and 'size' in meta:
                        total_size = meta['size']
                    print(f"|-- Downloading {filename} (Total size: {total_size} bytes)")
                    downloaded = 0
                    
                    with open(filepath, 'wb') as f:
                        for chunk in data_resp.iter_content(chunk_size=1024*1024):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    pct = (downloaded / total_size) * 100
                                    if pct > prev_pct + 5 or prev_pct == 0:
                                        prev_pct = pct
                                        print(f"| {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.0f}%)")
                    
                    if total_size > 0:
                        print()  # newline after progress
                
                print(f"[domain] Downloaded: {filename}")
                count += 1
                
            except Exception as e:
                print(f"[domain] Failed to download {data_id}: {e}")
        
        return count
    
    def upload_domain_data(self, domain_id: str, input_dir: Path) -> list[dict]:
        """
        Upload all files in a directory as domain data.
        
        Args:
            domain_id: Domain ID to upload to
            input_dir: Directory containing files to upload
        
        Returns:
            List of created domain data metadata
        """
        input_dir = Path(input_dir)
        results = []
        
        for filepath in input_dir.iterdir():
            if not filepath.is_file():
                continue
            
            filename = filepath.stem  # name without extension
            data_type = filepath.suffix.lstrip('.')
            
            if not data_type:
                print(f"[domain] Skipping {filepath.name} (no extension)")
                continue
            
            try:
                result = self.upload_file(domain_id, filepath, name=filename, data_type=data_type)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[domain] Failed to upload {filepath.name}: {e}")
        
        return results
    
    def upload_file(
        self,
        domain_id: str,
        filepath: Path,
        name: str,
        data_type: str,
        data_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Upload a single file as domain data.
        
        Args:
            domain_id: Domain ID to upload to
            filepath: Path to file to upload
            name: Name for the domain data
            data_type: Data type identifier
            data_id: Optional existing data ID (for updates via PUT)
        
        Returns:
            Created/updated domain data metadata, or None on failure
        """
        upload_url = f"{self.domain_server_url}/api/v1/domains/{domain_id}/data"
        http_method = "PUT" if data_id else "POST"
        
        print(f"[domain] Uploading {filepath.name} to {upload_url} ({http_method})")
        
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        part = {
            "name": name,
            "data_type": data_type,
            "domain_id": domain_id,
            "data": file_data,
        }
        if data_id:
            part["id"] = data_id
        
        body, content_type = build_multipart_body([part])
        
        headers = {
            "Content-Type": content_type,
            "Authorization": f"Bearer {self.access_token}",
        }
        
        if http_method == "POST":
            resp = requests.post(upload_url, data=body, headers=headers)
        else:
            resp = requests.put(upload_url, data=body, headers=headers)
        
        resp.raise_for_status()
        response_data = resp.json()
        
        # Response format: {"data": [{"id": "...", ...}]}
        if "data" in response_data and len(response_data["data"]) > 0:
            meta = response_data["data"][0]
            print(f"[domain] Uploaded: {name} -> {meta.get('id', 'unknown')}")
            return meta
        
        print(f"[domain] Uploaded {name} but no ID in response")
        return response_data


def get_client_from_env() -> Optional[DomainClient]:
    """Create domain client from environment variables."""
    domain_server_url = os.environ.get("DOMAIN_SERVER_URL")
    access_token = os.environ.get("DOMAIN_ACCESS_TOKEN")
    
    if not all([domain_server_url, access_token]):
        return None
    
    return DomainClient(domain_server_url, access_token)


#!/usr/bin/env python3
"""
convert_splat.py

Convert Gaussian splat files between formats:
  - PLY -> .splat (using ply2splat package)
  - .splat -> .sog (using splat-transform binary)

Usage:
    # Convert PLY to .splat
    python convert_splat.py input.ply --format splat
    
    # Convert PLY to both .splat and .sog
    python convert_splat.py input.ply --format sog
    
    # Convert existing .splat to .sog only
    python convert_splat.py input.splat --format sog
"""

import argparse
import subprocess
import sys
from pathlib import Path

import ply2splat

# Path to splat-transform binary for SOG conversion (Windows uses .cmd, Linux uses bare name)
SPLAT_TRANSFORM_BINARY = "splat-transform.cmd" if sys.platform == "win32" else "splat-transform"


def convert_ply_to_splat(input_path: Path, output_path: Path = None) -> Path:
    """
    Convert a PLY file to .splat format using ply2splat.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path for output .splat file (default: same name with .splat extension)
    
    Returns:
        Path to the created .splat file
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_suffix(".splat")
    
    print(f"Converting PLY to splat: {input_path.name} -> {output_path.name}")
    ply2splat.convert(str(input_path), str(output_path))
    
    return output_path


def convert_ply_to_sog(input_path: Path, output_path: Path = None) -> Path:
    """
    Convert a .ply file to .sog format using splat-transform.
    
    Args:
        input_path: Path to input .ply file
        output_path: Path for output .sog file (default: same name with .sog extension)
    
    Returns:
        Path to the created .sog file
    
    Raises:
        RuntimeError: If conversion fails
        FileNotFoundError: If splat-transform binary is not found
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_suffix(".sog")
    
    cmd = [SPLAT_TRANSFORM_BINARY, str(input_path), str(output_path)]
    
    print(f"Converting PLY to SOG: {input_path.name} -> {output_path.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SOG conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{SPLAT_TRANSFORM_BINARY} not found. Is it installed and in PATH?"
        )


def convert_file(input_path: Path, output_format: str) -> list[Path]:
    """
    Convert a splat file to the specified format(s).
    
    Args:
        input_path: Path to input file (.ply or .splat)
        output_format: Target format - "splat" or "sog"
    
    Returns:
        List of created output file paths
    """
    input_path = Path(input_path)
    input_ext = input_path.suffix.lower()
    outputs = []
    
    if input_ext != ".ply":
        raise ValueError(f"Input file must be a PLY file: {input_path}")

    if output_format == "splat":
        outputs.append(convert_ply_to_splat(input_path))
    elif output_format == "sog":
        outputs.append(convert_ply_to_sog(input_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Use 'splat' or 'sog'")
    
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gaussian splat files between PLY, .splat, and .sog formats."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (.ply or .splat)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["splat", "sog"],
        default="splat",
        help="Output format. 'sog' creates both .splat and .sog. Default: splat"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path. Default: same directory with appropriate extension"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        outputs = convert_file(args.input, args.format)
        print(f"Created: {', '.join(str(p) for p in outputs)}")
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

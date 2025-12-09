import ply2splat
import argparse

parser = argparse.ArgumentParser(description="Convert ply file to splat file")
parser.add_argument("--input", required=True, help="Input file")
parser.add_argument("--output", required=True, help="Output file")


args = parser.parse_args()

# Convert a PLY file to SPLAT format
count = ply2splat.convert(args.input, args.output)
print(f"Converted {count} splats, saved in: \n{args.output}")
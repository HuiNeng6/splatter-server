# splatter-server

## Build Docker
```bash
# on reoi root path
docker build -t splatter-server -f docker/Dockerfile .
```

## Run Trainer
```bash
python3 run.py \
--domain_id {domain_id}
--job_id {job_id} \
--job_root_path {path/to/job/root} \
--log_level {log level}
```

## Required Files
```bash
# Input Files
{job_root_path}
├── datasets
│   └── {dataset}
│       └── Frames.mp4
├── refined
│   └── global
│       └── refined_sfm_combined
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
```
## Output Files
```bash
# Output Files
{job_root_path}
├── Frames
│   ├── {images}
│   └── ... 
├── refined
│   ├── nerfstudio-data
│   │   └── {converted nerfstudio data from colmap}
│   └── splatter
│       ├── splat.ply
│       ├── splat_rot.ply
│       ├── splat_rot.splat # this is what needs to be uploaded to dmt
│       └── splatfacto
│           └── {splat torch model}
```
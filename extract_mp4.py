from pathlib import Path
import cv2
import os

def process_frames(
    scan_folder,
    output_path,
    every_nth_image,
):
    """
    Process and extract frames from video if necessary.
    
    Returns:
        Tuple of (reference_list, use_frames_from_video)
    """
    frames_mp4 = scan_folder / 'Frames.mp4'
    print(f"Looking for mp4 encoded frames: {frames_mp4}")
    
    use_frames_from_video = False
    if frames_mp4.exists():
        print(f"Frames mp4 found, unpacking into {output_path}")
        if not output_path.exists():
            output_path.mkdir()
        mp4_to_frames(frames_mp4, output_path, filename_prefix=f"{scan_folder.name}_")
        use_frames_from_video = True

    references = [str(p.relative_to(output_path)) for p in output_path.iterdir()]

    original_image_count = len(references)
    references = references[::every_nth_image]
    print(f'{len(references)}, frames selected, out of, {original_image_count}')
    return

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=Path, default="./datasets/dmt_scan_2024-06-26_10-26-52", help="Path to the input dataset"
    )
    parser.add_argument(
        "--output_path", type=Path, default="./outputs", help="Path for output files"
    )
    parser.add_argument(
        "--every_nth_image", type=int, default=1, help="Process every nth image"
    )
    args = parser.parse_args()

    process_frames(args.dataset_path, args.output_path, args.every_nth_image)
#!/usr/bin/env python3
"""
Convert aligned single-arm UMI raw data into a dataset.zarr.zip file that can be
read directly by diffusion_policy.dataset.umi_dataset.UmiDataset.

Expected raw layout for each episode:
    episodeX/
      camera/color/pikaFisheyeCamera/
        sync.txt
        *.jpg
      localization/pose/pika/
        sync.txt
        *.json
      gripper/encoder/pika/
        sync.txt
        *.json

The script writes a ReplayBuffer-compatible dataset to:
    /Users/sp/Desktop/umi_project/dataset/single/dataset.zarr.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
UMI_PROJECT_ROOT = REPO_ROOT / "universal_manipulation_interface"
if str(UMI_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(UMI_PROJECT_ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


DEFAULT_INPUT_ROOT = REPO_ROOT / "umidata" / "single"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "dataset" / "single" / "dataset.zarr.zip"
DEFAULT_TEST_OUTPUT_IMAGE = REPO_ROOT / "dataset" / "single" / "test_resize_output.jpg"

CAMERA_REL_PATH = Path("camera/color/pikaFisheyeCamera")
POSE_REL_PATH = Path("localization/pose/pika")
GRIPPER_REL_PATH = Path("gripper/encoder/pika")

OUTPUT_IMAGE_SIZE = (224, 224)  # width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert aligned single-arm UMI raw data to dataset.zarr.zip"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing episode* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output dataset.zarr.zip path.",
    )
    parser.add_argument(
        "--test-resize-image",
        type=Path,
        default=None,
        help="Optional path to a single RGB image for testing resize output only.",
    )
    parser.add_argument(
        "--test-output-image",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_IMAGE,
        help="Output path for the resized preview image when --test-resize-image is used.",
    )
    return parser.parse_args()


def episode_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.name
    suffix = stem.replace("episode", "")
    if suffix.isdigit():
        return (int(suffix), stem)
    return (10**9, stem)


def read_sync_file(sync_path: Path) -> List[str]:
    with sync_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def read_pose_json(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        pose_dict = json.load(f)

    xyz = np.array(
        [pose_dict["x"], pose_dict["y"], pose_dict["z"]],
        dtype=np.float32,
    )
    rpy = np.array(
        [pose_dict["roll"], pose_dict["pitch"], pose_dict["yaw"]],
        dtype=np.float32,
    )
    rotvec = Rotation.from_euler("xyz", rpy, degrees=False).as_rotvec().astype(
        np.float32
    )
    return np.concatenate([xyz, rotvec], axis=0)


def read_gripper_width(json_path: Path) -> float:
    with json_path.open("r", encoding="utf-8") as f:
        gripper_dict = json.load(f)
    return float(gripper_dict["distance"])


def read_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    ## crop + resize
    height, width = image_rgb.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped = image_rgb[top:top + crop_size, left:left + crop_size]
    resized = cv2.resize(
        cropped, OUTPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA
    )

    # # direct resize
    # resized = cv2.resize(
    #     image_rgb, OUTPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA
    # )
    return resized.astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), image_bgr)
    if not success:
        raise RuntimeError(f"Failed to save image to: {output_path}")


def run_resize_image_test(input_image: Path, output_image: Path) -> None:
    input_image = input_image.expanduser().resolve()
    output_image = output_image.expanduser().resolve()
    if not input_image.is_file():
        raise FileNotFoundError(f"Test image does not exist: {input_image}")

    resized_rgb = read_rgb_image(input_image)
    save_rgb_image(resized_rgb, output_image)

    print(f"Saved resized preview image to: {output_image}")
    print(f"Resized image shape: {resized_rgb.shape}")
    print(f"Resized image dtype: {resized_rgb.dtype}")


def build_episode_data(episode_dir: Path) -> dict:
    camera_dir = episode_dir / CAMERA_REL_PATH
    pose_dir = episode_dir / POSE_REL_PATH
    gripper_dir = episode_dir / GRIPPER_REL_PATH

    camera_sync = camera_dir / "sync.txt"
    pose_sync = pose_dir / "sync.txt"
    gripper_sync = gripper_dir / "sync.txt"

    missing_paths = [
        str(path)
        for path in (camera_sync, pose_sync, gripper_sync)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Episode {episode_dir.name} is missing required sync files: {missing_paths}"
        )

    camera_files = read_sync_file(camera_sync)
    pose_files = read_sync_file(pose_sync)
    gripper_files = read_sync_file(gripper_sync)

    if not camera_files or not pose_files or not gripper_files:
        raise ValueError(f"Episode {episode_dir.name} has empty sync.txt entries.")

    seq_len = min(len(camera_files), len(pose_files), len(gripper_files))
    if seq_len <= 0:
        raise ValueError(f"Episode {episode_dir.name} has no usable aligned frames.")

    if len({len(camera_files), len(pose_files), len(gripper_files)}) != 1:
        print(
            f"[WARN] {episode_dir.name}: sync length mismatch "
            f"(rgb={len(camera_files)}, pose={len(pose_files)}, gripper={len(gripper_files)}). "
            f"Using shortest length {seq_len}."
        )

    rgb_frames = []
    pose_series = []
    gripper_widths = []

    for idx in range(seq_len):
        image_path = camera_dir / camera_files[idx]
        pose_path = pose_dir / pose_files[idx]
        gripper_path = gripper_dir / gripper_files[idx]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing RGB file: {image_path}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")
        if not gripper_path.is_file():
            raise FileNotFoundError(f"Missing gripper file: {gripper_path}")

        rgb_frames.append(read_rgb_image(image_path))
        pose_series.append(read_pose_json(pose_path))
        gripper_widths.append(read_gripper_width(gripper_path))

    pose_array = np.stack(pose_series, axis=0).astype(np.float32)
    rgb_array = np.stack(rgb_frames, axis=0).astype(np.uint8)
    gripper_array = np.asarray(gripper_widths, dtype=np.float32).reshape(-1, 1)

    start_pose = np.repeat(pose_array[:1], repeats=seq_len, axis=0)
    end_pose = np.repeat(pose_array[-1:], repeats=seq_len, axis=0)

    return {
        "camera0_rgb": rgb_array,
        "robot0_eef_pos": pose_array[:, :3],
        "robot0_eef_rot_axis_angle": pose_array[:, 3:],
        "robot0_gripper_width": gripper_array,
        "robot0_demo_start_pose": start_pose,
        "robot0_demo_end_pose": end_pose,
    }


def main() -> None:
    args = parse_args()
    if args.test_resize_image is not None:
        run_resize_image_test(
            input_image=args.test_resize_image,
            output_image=args.test_output_image,
        )
        return

    input_root = args.input_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    episode_dirs = sorted(
        [path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("episode")],
        key=episode_sort_key,
    )
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {input_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    success_count = 0
    skipped = []

    print(f"Input root: {input_root}")
    print(f"Output path: {output_path}")
    print(f"Found {len(episode_dirs)} episode directories.")

    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        try:
            episode_data = build_episode_data(episode_dir)
            replay_buffer.add_episode(data=episode_data, compressors=None)
            success_count += 1
        except Exception as exc:
            skipped.append((episode_dir.name, str(exc)))
            print(f"[SKIP] {episode_dir.name}: {exc}")

    if success_count == 0:
        raise RuntimeError("No episodes were converted successfully.")

    if output_path.exists():
        output_path.unlink()

    with zarr.ZipStore(str(output_path), mode="w") as zip_store:
        replay_buffer.save_to_store(store=zip_store)

    print(f"Converted {success_count} episodes.")
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Saved dataset to: {output_path}")

    if skipped:
        print("Skipped episodes:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()

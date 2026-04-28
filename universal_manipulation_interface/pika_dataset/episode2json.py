#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class AlignedFrame:
    timestamp: float
    image_b64: str
    pose7_xyzw: list[float]
    gripper_width: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_episode_dir = repo_root / "pika_dataset" / "single" / "episode79"
    default_output_dir = repo_root / "pika_dataset" / "exports" / "episode79"

    parser = argparse.ArgumentParser(
        description=(
            "Read RGB / pose / gripper data from a Pika episode, align them by sync.txt, "
            "and export observation/action JSON pairs."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=default_episode_dir,
        help="Episode directory, e.g. pika_dataset/single/episode79",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="pikaFisheyeCamera",
        help="RGB camera directory name under camera/color/",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="pika",
        help="Pose device directory name under localization/pose/",
    )
    parser.add_argument(
        "--gripper-device",
        type=str,
        default="pika",
        help="Gripper device directory name under gripper/encoder/",
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="arm_l",
        help="Arm key used in payload_raw.json.",
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=2,
        help="Number of aligned frames stored in each observation payload.",
    )
    parser.add_argument(
        "--act-horizon",
        type=int,
        default=16,
        help="Number of aligned future frames stored in each actions.json.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride. Defaults to obs_horizon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory used to store exported samples.",
    )
    return parser.parse_args()


def read_sync_entries(sync_path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in sync_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"sync.txt is empty: {sync_path}")
    return entries


def encode_file_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("ascii")


def euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> list[float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]


def quat_xyzw_to_rotvec(quat_xyzw: Iterable[float]) -> list[float]:
    x, y, z, w = [float(value) for value in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    if w < 0.0:
        x = -x
        y = -y
        z = -z
        w = -w

    vec_norm = math.sqrt(x * x + y * y + z * z)
    if vec_norm < 1e-12:
        return [2.0 * x, 2.0 * y, 2.0 * z]

    angle = 2.0 * math.atan2(vec_norm, w)
    scale = angle / vec_norm
    return [x * scale, y * scale, z * scale]


def pose_json_to_pose7_xyzw(pose_json_path: Path) -> list[float]:
    raw_pose = json.loads(pose_json_path.read_text(encoding="utf-8"))
    position = [
        float(raw_pose["x"]),
        float(raw_pose["y"]),
        float(raw_pose["z"]),
    ]
    quat_xyzw = euler_xyz_to_quat_xyzw(
        roll=float(raw_pose["roll"]),
        pitch=float(raw_pose["pitch"]),
        yaw=float(raw_pose["yaw"]),
    )
    return position + quat_xyzw


def gripper_json_to_width(gripper_json_path: Path) -> float:
    raw_gripper = json.loads(gripper_json_path.read_text(encoding="utf-8"))
    return float(raw_gripper["distance"])


def pose7_xyzw_to_action(pose7_xyzw: Iterable[float], gripper_width: float) -> list[float]:
    pose7 = [float(x) for x in pose7_xyzw]
    if len(pose7) != 7:
        raise ValueError(f"Expected 7 values in pose7, got {len(pose7)}")
    position = pose7[:3]
    rotvec = quat_xyzw_to_rotvec(pose7[3:])
    return position + rotvec + [float(gripper_width)]


def build_aligned_frames(
    episode_dir: Path,
    camera_name: str,
    pose_device: str,
    gripper_device: str,
) -> list[AlignedFrame]:
    rgb_dir = episode_dir / "camera" / "color" / camera_name
    pose_dir = episode_dir / "localization" / "pose" / pose_device
    gripper_dir = episode_dir / "gripper" / "encoder" / gripper_device

    rgb_sync = read_sync_entries(rgb_dir / "sync.txt")
    pose_sync = read_sync_entries(pose_dir / "sync.txt")
    gripper_sync = read_sync_entries(gripper_dir / "sync.txt")

    aligned_count = min(len(rgb_sync), len(pose_sync), len(gripper_sync))
    if aligned_count == 0:
        raise ValueError("No aligned frames found from the three sync.txt files.")

    frames: list[AlignedFrame] = []
    for index in range(aligned_count):
        rgb_name = rgb_sync[index]
        pose_name = pose_sync[index]
        gripper_name = gripper_sync[index]

        rgb_path = rgb_dir / rgb_name
        pose_path = pose_dir / pose_name
        gripper_path = gripper_dir / gripper_name

        missing_paths = [path for path in (rgb_path, pose_path, gripper_path) if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing aligned files: {missing_text}")

        rgb_timestamp = float(Path(rgb_name).stem)
        pose_timestamp = float(Path(pose_name).stem)
        gripper_timestamp = float(Path(gripper_name).stem)
        aligned_timestamp = (rgb_timestamp + pose_timestamp + gripper_timestamp) / 3.0

        frames.append(
            AlignedFrame(
                timestamp=aligned_timestamp,
                image_b64=encode_file_base64(rgb_path),
                pose7_xyzw=pose_json_to_pose7_xyzw(pose_path),
                gripper_width=gripper_json_to_width(gripper_path),
            )
        )
    return frames


def build_observation_payload(
    arm_name: str,
    init_pose: list[float],
    obs_frames: list[AlignedFrame],
) -> dict:
    arm_payload = {
        "images": [frame.image_b64 for frame in obs_frames],
        "init_pose": init_pose,
        "poses": [frame.pose7_xyzw for frame in obs_frames],
        "grippers": [frame.gripper_width for frame in obs_frames],
        "timestamps": [frame.timestamp for frame in obs_frames],
    }
    return {
        arm_name: arm_payload,
        "send_timestamp": obs_frames[-1].timestamp,
        "type": "observation",
    }


def build_action_payload(action_frames: list[AlignedFrame]) -> dict:
    return {
        "actions": [
            pose7_xyzw_to_action(frame.pose7_xyzw, frame.gripper_width)
            for frame in action_frames
        ]
    }


def export_samples(
    aligned_frames: list[AlignedFrame],
    arm_name: str,
    obs_horizon: int,
    act_horizon: int,
    stride: int,
    output_dir: Path,
) -> int:
    if obs_horizon <= 0:
        raise ValueError("obs_horizon must be positive.")
    if act_horizon <= 0:
        raise ValueError("act_horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    total_needed = obs_horizon + act_horizon
    if len(aligned_frames) < total_needed:
        raise ValueError(
            f"Aligned frames are not enough. Need at least {total_needed}, got {len(aligned_frames)}."
        )

    init_pose = aligned_frames[0].pose7_xyzw
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    max_start = len(aligned_frames) - total_needed
    for start_idx in range(0, max_start + 1, stride):
        obs_frames = aligned_frames[start_idx : start_idx + obs_horizon]
        action_frames = aligned_frames[
            start_idx + obs_horizon : start_idx + obs_horizon + act_horizon
        ]
        sample_dir = output_dir / f"sample_{sample_count:06d}"
        sample_dir.mkdir(parents=True, exist_ok=False)

        observation_payload = build_observation_payload(
            arm_name=arm_name,
            init_pose=init_pose,
            obs_frames=obs_frames,
        )
        action_payload = build_action_payload(action_frames)

        (sample_dir / "payload_raw.json").write_text(
            json.dumps(observation_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "actions.json").write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "meta.json").write_text(
            json.dumps(
                {
                    "start_index": start_idx,
                    "obs_horizon": obs_horizon,
                    "act_horizon": act_horizon,
                    "stride": stride,
                    "obs_timestamps": [frame.timestamp for frame in obs_frames],
                    "action_timestamps": [frame.timestamp for frame in action_frames],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        sample_count += 1

    return sample_count


def main() -> None:
    args = parse_args()
    stride = args.obs_horizon if args.stride is None else args.stride

    aligned_frames = build_aligned_frames(
        episode_dir=args.episode_dir,
        camera_name=args.camera_name,
        pose_device=args.pose_device,
        gripper_device=args.gripper_device,
    )
    sample_count = export_samples(
        aligned_frames=aligned_frames,
        arm_name=args.arm_name,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        stride=stride,
        output_dir=args.output_dir,
    )

    summary = {
        "episode_dir": str(args.episode_dir),
        "output_dir": str(args.output_dir),
        "aligned_frame_count": len(aligned_frames),
        "obs_horizon": args.obs_horizon,
        "act_horizon": args.act_horizon,
        "stride": stride,
        "sample_count": sample_count,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import DEFAULT_CHECKPOINT_PATH, infer_payload, load_inference_context


DEFAULT_SAMPLES_DIR = Path(
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/pika_dataset/exports/episode79"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate/eval_output"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse exported open-loop samples, run inference on each payload_raw.json, "
            "and compare the predicted env_action with actions.json."
        )
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory containing sample_*/payload_raw.json and actions.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint file or checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save comparison results",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="arm_l",
        choices=["arm_l", "arm_r"],
        help="Arm payload key used for inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def list_sample_dirs(samples_dir: Path) -> list[Path]:
    sample_dirs = sorted(
        path for path in samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    )
    if not sample_dirs:
        raise ValueError(f"No sample_* directories found in {samples_dir}")
    return sample_dirs


def select_arm_payload(payload: dict, arm: str) -> dict:
    if "images" in payload and "poses" in payload and "grippers" in payload:
        return payload
    if arm in payload:
        return payload[arm]
    available_arms = [key for key in ("arm_l", "arm_r") if key in payload]
    raise ValueError(f"Cannot find arm `{arm}` in payload. Available arms: {available_arms}")


def validate_payload_pose_format(payload: dict, arm: str) -> dict:
    arm_payload = select_arm_payload(payload, arm)
    poses = np.asarray(arm_payload.get("poses", []), dtype=np.float64)
    init_pose = np.asarray(arm_payload.get("init_pose", []), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected payload poses to have shape [T, 7], got {poses.shape}")
    if init_pose.shape != (7,):
        raise ValueError(f"Expected payload init_pose to have shape [7], got {init_pose.shape}")

    quat_norms = np.linalg.norm(poses[:, 3:], axis=1)
    init_quat_norm = float(np.linalg.norm(init_pose[3:]))
    return {
        "format": "xyz_quat_xyzw",
        "pose_shape": list(poses.shape),
        "init_pose_shape": list(init_pose.shape),
        "quat_norm_mean": float(quat_norms.mean()),
        "quat_norm_min": float(quat_norms.min()),
        "quat_norm_max": float(quat_norms.max()),
        "init_quat_norm": init_quat_norm,
    }


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    small = angle < 1e-12

    scale = np.empty_like(angle)
    scale[~small] = np.sin(half_angle[~small]) / angle[~small]
    scale[small] = 0.5 - (angle[small] ** 2) / 48.0

    quat_xyz = rotvec * scale
    quat_w = np.cos(half_angle)
    quat = np.concatenate([quat_xyz, quat_w], axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat


def quat_conjugate(quat_xyzw: np.ndarray) -> np.ndarray:
    out = quat_xyzw.copy()
    out[..., :3] *= -1.0
    return out


def quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(quat_a, -1, 0)
    bx, by, bz, bw = np.moveaxis(quat_b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def geodesic_angle_from_rotvecs(pred_rotvec: np.ndarray, target_rotvec: np.ndarray) -> np.ndarray:
    pred_quat = rotvec_to_quat_xyzw(pred_rotvec)
    target_quat = rotvec_to_quat_xyzw(target_rotvec)
    rel_quat = quat_multiply(pred_quat, quat_conjugate(target_quat))
    rel_quat /= np.linalg.norm(rel_quat, axis=-1, keepdims=True)
    xyz_norm = np.linalg.norm(rel_quat[:, :3], axis=-1)
    w_abs = np.abs(rel_quat[:, 3])
    return 2.0 * np.arctan2(xyz_norm, w_abs)


def compute_diff_metrics(pred_action: np.ndarray, target_action: np.ndarray) -> dict:
    if pred_action.ndim != 2 or target_action.ndim != 2:
        raise ValueError(
            f"Expected 2D actions, got pred shape {pred_action.shape}, target shape {target_action.shape}"
        )

    common_steps = min(pred_action.shape[0], target_action.shape[0])
    common_dims = min(pred_action.shape[1], target_action.shape[1])
    pred = pred_action[:common_steps, :common_dims]
    target = target_action[:common_steps, :common_dims]
    if common_dims < 7:
        raise ValueError(
            f"Expected at least 7 action dims for xyz + rotvec + gripper, got {common_dims}"
        )

    diff = pred[:, :7] - target[:, :7]
    position_diff = diff[:, :3]
    gripper_abs_error = np.abs(diff[:, 6])
    position_l2 = np.linalg.norm(position_diff, axis=1)
    rotation_geodesic_rad = geodesic_angle_from_rotvecs(pred[:, 3:6], target[:, 3:6])
    rotation_geodesic_deg = np.rad2deg(rotation_geodesic_rad)

    return {
        "aligned_shape": [int(common_steps), 7],
        "pred_shape": list(pred_action.shape),
        "target_shape": list(target_action.shape),
        "position": {
            "mean_l2_error": float(position_l2.mean()),
            "rmse_l2_error": float(np.sqrt(np.mean(position_l2 ** 2))),
            "max_l2_error": float(position_l2.max()),
            "per_axis_mae": np.abs(position_diff).mean(axis=0).tolist(),
            "per_axis_rmse": np.sqrt(np.mean(position_diff ** 2, axis=0)).tolist(),
        },
        "rotation": {
            "mean_geodesic_rad": float(rotation_geodesic_rad.mean()),
            "rmse_geodesic_rad": float(np.sqrt(np.mean(rotation_geodesic_rad ** 2))),
            "max_geodesic_rad": float(rotation_geodesic_rad.max()),
            "mean_geodesic_deg": float(rotation_geodesic_deg.mean()),
            "rmse_geodesic_deg": float(np.sqrt(np.mean(rotation_geodesic_deg ** 2))),
            "max_geodesic_deg": float(rotation_geodesic_deg.max()),
        },
        "gripper": {
            "mean_abs_error": float(gripper_abs_error.mean()),
            "rmse": float(np.sqrt(np.mean(gripper_abs_error ** 2))),
            "max_abs_error": float(gripper_abs_error.max()),
        },
        "combined_legacy": {
            "mean_abs_error": float(np.abs(diff).mean()),
            "max_abs_error": float(np.abs(diff).max()),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
        },
        "pred_action_aligned": pred[:, :7].tolist(),
        "target_action_aligned": target[:, :7].tolist(),
        "diff_aligned": diff.tolist(),
    }


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    sample_dirs = list_sample_dirs(args.samples_dir)
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = args.output_dir / "per_sample"
    context = load_inference_context(args.checkpoint)

    sample_results: list[dict] = []
    position_mean_values: list[float] = []
    position_rmse_values: list[float] = []
    position_max_values: list[float] = []
    rotation_mean_deg_values: list[float] = []
    rotation_rmse_deg_values: list[float] = []
    rotation_max_deg_values: list[float] = []
    gripper_mean_values: list[float] = []
    gripper_rmse_values: list[float] = []
    gripper_max_values: list[float] = []
    pose_validation_results: list[dict] = []

    for sample_dir in sample_dirs:
        payload_path = sample_dir / "payload_raw.json"
        actions_path = sample_dir / "actions.json"
        meta_path = sample_dir / "meta.json"

        payload = load_json(payload_path)
        target_actions = load_json(actions_path)
        meta = load_json(meta_path) if meta_path.exists() else {}
        payload_pose_validation = validate_payload_pose_format(payload, args.arm)
        pose_validation_results.append(payload_pose_validation)

        inference_output = infer_payload(payload=payload, context=context, arm=args.arm)
        pred_action = np.asarray(inference_output["env_action"], dtype=np.float32)
        target_action = np.asarray(target_actions["actions"], dtype=np.float32)
        metrics = compute_diff_metrics(pred_action=pred_action, target_action=target_action)

        sample_result = {
            "sample_name": sample_dir.name,
            "payload_path": str(payload_path),
            "actions_path": str(actions_path),
            "meta": meta,
            "payload_pose_validation": payload_pose_validation,
            "inference": {
                "selected_arm": inference_output["selected_arm"],
                "checkpoint_path": inference_output["checkpoint_path"],
                "device": inference_output["device"],
                "action_pred_shape": inference_output["action_pred_shape"],
                "env_action_shape": inference_output["env_action_shape"],
            },
            "metrics": metrics,
        }
        save_json(sample_result, per_sample_dir / f"{sample_dir.name}.json")

        sample_results.append(
            {
                "sample_name": sample_dir.name,
                "position_mean_l2_error": metrics["position"]["mean_l2_error"],
                "position_rmse_l2_error": metrics["position"]["rmse_l2_error"],
                "position_max_l2_error": metrics["position"]["max_l2_error"],
                "rotation_mean_geodesic_deg": metrics["rotation"]["mean_geodesic_deg"],
                "rotation_rmse_geodesic_deg": metrics["rotation"]["rmse_geodesic_deg"],
                "rotation_max_geodesic_deg": metrics["rotation"]["max_geodesic_deg"],
                "gripper_mean_abs_error": metrics["gripper"]["mean_abs_error"],
                "gripper_rmse": metrics["gripper"]["rmse"],
                "gripper_max_abs_error": metrics["gripper"]["max_abs_error"],
                "aligned_shape": metrics["aligned_shape"],
            }
        )
        position_mean_values.append(metrics["position"]["mean_l2_error"])
        position_rmse_values.append(metrics["position"]["rmse_l2_error"])
        position_max_values.append(metrics["position"]["max_l2_error"])
        rotation_mean_deg_values.append(metrics["rotation"]["mean_geodesic_deg"])
        rotation_rmse_deg_values.append(metrics["rotation"]["rmse_geodesic_deg"])
        rotation_max_deg_values.append(metrics["rotation"]["max_geodesic_deg"])
        gripper_mean_values.append(metrics["gripper"]["mean_abs_error"])
        gripper_rmse_values.append(metrics["gripper"]["rmse"])
        gripper_max_values.append(metrics["gripper"]["max_abs_error"])

    summary = {
        "samples_dir": str(args.samples_dir),
        "checkpoint_path": str(args.checkpoint),
        "arm": args.arm,
        "sample_count": len(sample_results),
        "payload_pose_format": {
            "assumed_format": "xyz_quat_xyzw",
            "validated_pose_shape": pose_validation_results[0]["pose_shape"],
            "validated_init_pose_shape": pose_validation_results[0]["init_pose_shape"],
            "quat_norm_mean_across_samples": float(np.mean([x["quat_norm_mean"] for x in pose_validation_results])),
            "quat_norm_min_across_samples": float(np.min([x["quat_norm_min"] for x in pose_validation_results])),
            "quat_norm_max_across_samples": float(np.max([x["quat_norm_max"] for x in pose_validation_results])),
        },
        "global_metrics": {
            "position": {
                "mean_of_sample_mean_l2_error": float(np.mean(position_mean_values)),
                "mean_of_sample_rmse_l2_error": float(np.mean(position_rmse_values)),
                "max_of_sample_max_l2_error": float(np.max(position_max_values)),
            },
            "rotation": {
                "mean_of_sample_mean_geodesic_deg": float(np.mean(rotation_mean_deg_values)),
                "mean_of_sample_rmse_geodesic_deg": float(np.mean(rotation_rmse_deg_values)),
                "max_of_sample_max_geodesic_deg": float(np.max(rotation_max_deg_values)),
            },
            "gripper": {
                "mean_of_sample_mean_abs_error": float(np.mean(gripper_mean_values)),
                "mean_of_sample_rmse": float(np.mean(gripper_rmse_values)),
                "max_of_sample_max_abs_error": float(np.max(gripper_max_values)),
            },
        },
        "samples": sample_results,
    }
    save_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

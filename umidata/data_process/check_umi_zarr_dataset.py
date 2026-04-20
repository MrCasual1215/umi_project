#!/usr/bin/env python3
"""
Quick structural validator for single-arm UMI dataset.zarr.zip files.

This script checks whether the generated zarr dataset matches the core
expectations of diffusion_policy.dataset.umi_dataset.UmiDataset together with
the default single-arm task config in diffusion_policy/config/task/umi.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "single" / "dataset.zarr.zip"

REQUIRED_KEYS = {
    "camera0_rgb": {
        "shape_suffix": (224, 224, 3),
        "dtype_kind": ("u",),
        "dtype_desc": "uint8-like",
    },
    "robot0_eef_pos": {
        "shape_suffix": (3,),
        "dtype_kind": ("f",),
        "dtype_desc": "floating-point",
    },
    "robot0_eef_rot_axis_angle": {
        "shape_suffix": (3,),
        "dtype_kind": ("f",),
        "dtype_desc": "floating-point",
    },
    "robot0_gripper_width": {
        "shape_suffix": (1,),
        "dtype_kind": ("f",),
        "dtype_desc": "floating-point",
    },
    "robot0_demo_start_pose": {
        "shape_suffix": (6,),
        "dtype_kind": ("f",),
        "dtype_desc": "floating-point",
    },
    "robot0_demo_end_pose": {
        "shape_suffix": (6,),
        "dtype_kind": ("f",),
        "dtype_desc": "floating-point",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a dataset.zarr.zip file matches single-arm UMI expectations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to dataset.zarr.zip",
    )
    return parser.parse_args()


def array_info(arr: zarr.Array) -> Dict[str, object]:
    return {
        "shape": tuple(arr.shape),
        "dtype": np.dtype(arr.dtype),
    }


def validate_episode_ends(episode_ends: np.ndarray, expected_total_steps: int) -> List[str]:
    errors: List[str] = []
    if episode_ends.ndim != 1:
        errors.append(f"`meta/episode_ends` must be 1D, got shape {episode_ends.shape}.")
        return errors
    if len(episode_ends) == 0:
        errors.append("`meta/episode_ends` must not be empty.")
        return errors
    if not np.issubdtype(episode_ends.dtype, np.integer):
        errors.append(
            f"`meta/episode_ends` must be integer typed, got dtype {episode_ends.dtype}."
        )
    diffs = np.diff(episode_ends)
    if np.any(diffs <= 0):
        errors.append("`meta/episode_ends` must be strictly increasing.")
    if int(episode_ends[-1]) != expected_total_steps:
        errors.append(
            f"`meta/episode_ends[-1]` must equal total step count {expected_total_steps}, "
            f"got {int(episode_ends[-1])}."
        )
    return errors


def validate_required_arrays(
    data_group: zarr.Group,
) -> Tuple[List[str], Dict[str, Dict[str, object]], int]:
    errors: List[str] = []
    infos: Dict[str, Dict[str, object]] = {}
    total_steps: int | None = None

    for key, spec in REQUIRED_KEYS.items():
        if key not in data_group:
            errors.append(f"Missing required dataset key: `data/{key}`.")
            continue

        arr = data_group[key]
        info = array_info(arr)
        infos[key] = info
        shape = info["shape"]
        dtype = info["dtype"]

        if len(shape) < 2:
            errors.append(f"`data/{key}` must have time dimension plus feature dims, got {shape}.")
            continue

        this_total_steps = shape[0]
        if total_steps is None:
            total_steps = this_total_steps
        elif this_total_steps != total_steps:
            errors.append(
                f"Time dimension mismatch for `data/{key}`: expected {total_steps}, got {this_total_steps}."
            )

        expected_suffix = spec["shape_suffix"]
        if tuple(shape[1:]) != expected_suffix:
            errors.append(
                f"`data/{key}` must have trailing shape {expected_suffix}, got {shape[1:]}."
            )

        if dtype.kind not in spec["dtype_kind"]:
            errors.append(
                f"`data/{key}` must be {spec['dtype_desc']}, got dtype {dtype}."
            )

    return errors, infos, total_steps if total_steps is not None else 0


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()

    if not input_path.is_file():
        print("FAIL")
        print(f"- Dataset file does not exist: {input_path}")
        return 1

    errors: List[str] = []
    infos: Dict[str, Dict[str, object]] = {}
    episode_count = 0
    total_steps = 0

    try:
        with zarr.ZipStore(str(input_path), mode="r") as store:
            root = zarr.group(store=store)

            if "meta" not in root:
                errors.append("Missing root group: `meta`.")
            if "data" not in root:
                errors.append("Missing root group: `data`.")

            if not errors:
                meta_group = root["meta"]
                data_group = root["data"]

                if "episode_ends" not in meta_group:
                    errors.append("Missing required metadata array: `meta/episode_ends`.")
                else:
                    episode_ends = meta_group["episode_ends"][:]
                    episode_count = int(len(episode_ends))

                array_errors, infos, total_steps = validate_required_arrays(data_group)
                errors.extend(array_errors)

                if "episode_ends" in meta_group and total_steps > 0:
                    episode_ends = meta_group["episode_ends"][:]
                    errors.extend(validate_episode_ends(episode_ends, total_steps))

    except Exception as exc:
        print("FAIL")
        print(f"- Failed to open or inspect zarr dataset: {exc}")
        return 1

    if errors:
        print("FAIL")
        print(f"- Input: {input_path}")
        for error in errors:
            print(f"- {error}")
        if infos:
            print("- Observed arrays:")
            for key, info in infos.items():
                print(f"- data/{key}: shape={info['shape']}, dtype={info['dtype']}")
        return 1

    print("PASS")
    print(f"- Input: {input_path}")
    print(f"- Episodes: {episode_count}")
    print(f"- Total steps: {total_steps}")
    for key, info in infos.items():
        print(f"- data/{key}: shape={info['shape']}, dtype={info['dtype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

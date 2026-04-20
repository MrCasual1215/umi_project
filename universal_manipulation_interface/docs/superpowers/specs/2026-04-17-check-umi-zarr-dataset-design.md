# Check UMI Zarr Dataset Design

## Goal

Add a lightweight Python script that quickly validates whether a generated
`dataset.zarr.zip` matches the structural expectations of
`diffusion_policy.dataset.umi_dataset.UmiDataset` for the single-arm UMI task.

This script is intended for fast pre-training checks, not for full end-to-end
dataset loading.

## Scope

The script will:

- Open a `dataset.zarr.zip` file with `zarr.ZipStore`
- Verify the presence of `meta/episode_ends`
- Verify required `data/*` arrays for the single-arm UMI path:
  - `camera0_rgb`
  - `robot0_eef_pos`
  - `robot0_eef_rot_axis_angle`
  - `robot0_gripper_width`
  - `robot0_demo_start_pose`
  - `robot0_demo_end_pose`
- Check that all arrays share the same time dimension `T`
- Check that `episode_ends` is strictly increasing and ends at `T`
- Check expected shapes:
  - `camera0_rgb`: `(T, 224, 224, 3)`
  - `robot0_eef_pos`: `(T, 3)`
  - `robot0_eef_rot_axis_angle`: `(T, 3)`
  - `robot0_gripper_width`: `(T, 1)`
  - `robot0_demo_start_pose`: `(T, 6)`
  - `robot0_demo_end_pose`: `(T, 6)`
- Check dtypes:
  - `camera0_rgb` should be `uint8`
  - low-dimensional arrays should be floating-point
- Print a concise PASS / FAIL summary and key metadata

The script will not:

- Instantiate `UmiDataset`
- Run sequence sampling
- Validate task-specific latency or horizon settings
- Check semantic correctness of pose values

## Recommended Location

Place the script at:

`/Users/sp/Desktop/umi_project/umidata/data_process/check_umi_zarr_dataset.py`

## Interface

Default usage:

```bash
python3 /Users/sp/Desktop/umi_project/umidata/data_process/check_umi_zarr_dataset.py
```

Optional override:

```bash
python3 /Users/sp/Desktop/umi_project/umidata/data_process/check_umi_zarr_dataset.py \
  --input /Users/sp/Desktop/umi_project/dataset/single/dataset.zarr.zip
```

## Output

On success:

- Print `PASS`
- Print episode count, total step count, and each required key's shape / dtype

On failure:

- Print `FAIL`
- Print each violated condition in a readable list
- Exit with non-zero status

## Validation Logic

1. Open the zip-backed zarr store.
2. Verify `meta/episode_ends` exists and is one-dimensional.
3. Verify required arrays exist under `data/`.
4. Read metadata for each array and compare first dimension lengths.
5. Validate expected trailing dimensions and dtypes.
6. Validate `episode_ends`:
   - non-empty
   - strictly increasing
   - last value equals total frame count
7. Emit summary and exit status.

## Risks and Decisions

- The script is intentionally task-specific to keep it simple and useful for
  your current single-arm pipeline.
- It checks structural compatibility only; a dataset can pass this script and
  still contain poor pose conventions or incorrect synchronization.
- The image shape check is pinned to `224x224` because the current conversion
  script resizes to that format and `task/umi.yaml` expects `[3, 224, 224]`.

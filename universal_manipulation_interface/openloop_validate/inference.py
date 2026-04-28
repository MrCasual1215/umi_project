import argparse
import base64
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import time
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_action,
    get_real_umi_obs_dict,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)

try:
    import dill as torch_pickle_module
except ImportError:
    torch_pickle_module = pickle


DEFAULT_PAYLOAD_PATH = Path(
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/realworld_deploy/output/received_observations/20260423/obs_20260423_162745_1776932865640826015/payload_raw.json"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/"
    "data/outputs/2026.04.20/16.30.06_train_diffusion_unet_timm_picknplace/checkpoints"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/inference_output/output.json"
)


@dataclass
class InferenceContext:
    checkpoint_path: Path
    ckpt_payload: dict
    cfg: object
    device: torch.device
    policy: object


def decode_rgb_image(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64, validate=True)
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("Failed to decode base64 JPEG image.")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def crop_resize_rgb_image(image_rgb: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped = image_rgb[top:top + crop_size, left:left + crop_size]
    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), image_bgr)
    if not success:
        raise ValueError(f"Failed to save image to {output_path}")


def pose7_to_pos_axis_angle(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose7.shape}")
    pos = pose7[:3]
    quat_xyzw = pose7[3:]
    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return np.concatenate([pos, rotvec], axis=0)


def build_env_obs(
    payload: dict,
    shape_meta: dict,
    image_output_dir: Optional[Path] = None,
) -> tuple[dict, list[np.ndarray]]:
    obs_shape_meta = shape_meta["obs"]
    # print(f"obs_shape_meta: {obs_shape_meta}")
    env_obs: dict[str, np.ndarray] = {}

    decoded_images = [decode_rgb_image(image_b64) for image_b64 in payload.get("images", [])]
    # print(f"shape of decoded_images: {len(decoded_images)}, {decoded_images[0].shape}")
    # current shape:  2, (480, 640, 3)

    poses = [pose7_to_pos_axis_angle(np.asarray(pose)) for pose in payload.get("poses", [])]
    # print(f"shape of poses: {len(poses)}, {poses[0].shape}")
    # current shape:  2, (6,)

    grippers = [float(x) for x in payload.get("grippers", [])]
    # print(f"shape of grippers: {len(grippers)}")
    # current shape:  2

    robot_indices = sorted(
        {
            int(key.split("_")[0].replace("robot", ""))
            for key in obs_shape_meta.keys()
            if key.startswith("robot")
        }
    )
    # print(f"robot_indices: {robot_indices}")
    # [0]

    image_keys = sorted(
        [key for key, attr in obs_shape_meta.items() if attr.get("type", "low_dim") == "rgb"]
    )
    # print(f"image_keys: {image_keys}")
    # ['camera0_rgb']

    if len(decoded_images) < len(image_keys):
        raise ValueError(
            f"Payload only has {len(decoded_images)} images, but shape_meta requires at least {len(image_keys)}."
        )
    if len(poses) == 0:
        raise ValueError(
            "Payload does not contain any pose history."
        )
    if len(grippers) == 0:
        raise ValueError(
            "Payload does not contain any gripper history."
        )

    for key in image_keys:
        # print(f"key: {key}")
        # print(f"image_idx: {image_idx}")
        horizon = int(obs_shape_meta[key]["horizon"]) # 2
        if len(decoded_images) < horizon:
            raise ValueError(
                f"Payload only has {len(decoded_images)} image frames, but {key} requires horizon={horizon}."
            )
        co, ho, wo = obs_shape_meta[key]["shape"]
        if co != 3:
            raise ValueError(f"Expected 3-channel RGB image for {key}, got shape_meta={obs_shape_meta[key]['shape']}")
        image_hist = decoded_images[-horizon:]
        image_shapes = [img.shape for img in image_hist]
        if len(set(image_shapes)) != 1:
            raise ValueError(f"Inconsistent image shapes in history for {key}: {image_shapes}")
        processed_images = [
            crop_resize_rgb_image(image_rgb=img, output_size=(wo, ho))
            for img in image_hist
        ]
        env_obs[key] = np.stack(processed_images, axis=0)

        if image_output_dir is not None:
            saved_paths = []
            for frame_idx, image_rgb in enumerate(processed_images):
                image_path = image_output_dir / f"{key}_t{frame_idx:02d}.png"
                save_rgb_image(image_rgb, image_path)
                saved_paths.append(str(image_path))
            print(f"saved processed images for {key}: {saved_paths}")

    init_pose_raw = payload.get("init_pose")
    if init_pose_raw is None:
        init_pose = poses[0].copy()
        print("warning: init_pose is None, using first pose in poses as init_pose.")
    else:
        init_pose = pose7_to_pos_axis_angle(np.asarray(init_pose_raw))

    episode_start_pose = [init_pose]
    for robot_idx in robot_indices:
        pos_key = f"robot{robot_idx}_eef_pos"
        rot_key = f"robot{robot_idx}_eef_rot_axis_angle"
        grip_key = f"robot{robot_idx}_gripper_width"

        if pos_key in obs_shape_meta:
            horizon = int(obs_shape_meta[pos_key]["horizon"])
            if len(poses) < horizon:
                raise ValueError(
                    f"Payload only has {len(poses)} pose frames, but {pos_key} requires horizon={horizon}."
                )
            pose_hist = np.stack(poses[-horizon:], axis=0).astype(np.float32)
            env_obs[pos_key] = pose_hist[:, :3]
        if rot_key in obs_shape_meta:
            horizon = int(obs_shape_meta[rot_key]["horizon"])
            if len(poses) < horizon:
                raise ValueError(
                    f"Payload only has {len(poses)} pose frames, but {rot_key} requires horizon={horizon}."
                )
            pose_hist = np.stack(poses[-horizon:], axis=0).astype(np.float32)
            env_obs[rot_key] = pose_hist[:, 3:]
        if grip_key in obs_shape_meta:
            horizon = int(obs_shape_meta[grip_key]["horizon"])
            if len(grippers) < horizon:
                raise ValueError(
                    f"Payload only has {len(grippers)} gripper frames, but {grip_key} requires horizon={horizon}."
                )
            env_obs[grip_key] = np.asarray(grippers[-horizon:], dtype=np.float32)[:, None]

    return env_obs, episode_start_pose


def load_checkpoint(ckpt_path: Path) -> tuple[dict, object]:
    resolved_ckpt_path = ckpt_path
    if ckpt_path.is_dir():
        resolved_ckpt_path = ckpt_path / "latest.ckpt"
    payload = torch.load(
        open(resolved_ckpt_path, "rb"),
        map_location="cpu",
        pickle_module=torch_pickle_module,
    )
    return payload, payload["cfg"]


def create_policy(cfg, payload: dict, device: torch.device):
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.num_inference_steps = 16
    policy.eval().to(device)
    policy.reset()
    return policy


def load_inference_context(checkpoint_path: Path) -> InferenceContext:
    ckpt_payload, cfg = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = create_policy(cfg, ckpt_payload, device)
    return InferenceContext(
        checkpoint_path=checkpoint_path,
        ckpt_payload=ckpt_payload,
        cfg=cfg,
        device=device,
        policy=policy,
    )


def select_arm_payload(payload: dict, arm: str) -> tuple[str, dict]:
    if "images" in payload and "poses" in payload and "grippers" in payload:
        return arm, payload

    available_arms = [key for key in ("arm_l", "arm_r") if key in payload]
    if not available_arms:
        raise ValueError(
            "Payload format is not recognized. Expected flat keys "
            "(`images`, `poses`, `grippers`) or nested `arm_l` / `arm_r`."
        )
    if len(available_arms) == 1:
        selected_arm = available_arms[0]
        print(f"Only one arm found in payload, automatically using `{selected_arm}`.")
        return selected_arm, payload[selected_arm]
    if arm not in payload:
        raise ValueError(
            f"Requested arm `{arm}` not found in payload. Available arms: {available_arms}"
        )
    return arm, payload[arm]


def infer_payload(
    payload: dict,
    context: InferenceContext,
    arm: str,
) -> dict:
    selected_arm, arm_payload = select_arm_payload(payload, arm)
    cfg = context.cfg
    device = context.device

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    env_obs, episode_start_pose = build_env_obs(
        arm_payload,
        cfg.task.shape_meta,
        image_output_dir=None,
    )

    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    obs_dict_np = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=cfg.task.shape_meta,
        obs_pose_repr=obs_pose_repr,
        tx_robot1_robot0=np.eye(4, dtype=np.float32),
        episode_start_pose=episode_start_pose,
    )
    obs_dict = dict_apply(
        obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        start_time = time.time()
        context.policy.reset()
        result = context.policy.predict_action(obs_dict)
        end_time = time.time()
        print(f"inference time: {end_time - start_time}")

    raw_action = result["action_pred"][0].detach().cpu().numpy()
    env_action = get_real_umi_action(raw_action, env_obs, action_pose_repr)

    output = {
        "selected_arm": selected_arm,
        "checkpoint_path": str(context.checkpoint_path),
        "resolved_obs_resolution": list(obs_res),
        "device": str(device),
        "obs_pose_repr": obs_pose_repr,
        "action_pose_repr": action_pose_repr,
        "input_summary": {
            "num_images": len(arm_payload.get("images", [])),
            "num_poses": len(arm_payload.get("poses", [])),
            "num_grippers": len(arm_payload.get("grippers", [])),
        },
        "obs_tensor_shapes": {
            key: list(value.shape) for key, value in obs_dict_np.items()
        },
        "action_pred_shape": list(raw_action.shape),
        "action_pred": raw_action.tolist(),
        "env_action_shape": list(env_action.shape),
        "env_action": env_action.tolist(),
    }
    return output


def run_inference(
    payload_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    arm: str,
) -> Path:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    context = load_inference_context(checkpoint_path)
    output = infer_payload(payload=payload, context=context, arm=arm)
    output["payload_path"] = str(payload_path)
    output["processed_image_dir"] = str(output_path.parent / "processed_images" / output["selected_arm"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal offline policy inference from a payload_raw.json file."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=DEFAULT_PAYLOAD_PATH,
        help="Path to payload_raw.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint file or checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save inference result json",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="arm_l",
        choices=["arm_l", "arm_r"],
        help="Which arm payload to use when payload_raw.json contains both arms.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = run_inference(
        payload_path=args.payload,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        arm=args.arm,
    )
    print(f"Saved inference result to: {output_path}")


if __name__ == "__main__":
    main()

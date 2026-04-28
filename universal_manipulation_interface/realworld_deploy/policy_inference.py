import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_action,
    get_real_umi_obs_dict,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class PolicyInference:
    def __init__(
        self,
        checkpoint_path: str,
        preferred_arm: str = "arm_l",
        device: Optional[str] = None,
        verbose: bool = True,
        _print: bool = True,
        img_save: bool = False,
        crop: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.preferred_arm = preferred_arm
        self.verbose = verbose
        self.img_save = img_save
        self.print = _print
        self.crop = crop

        self.ckpt_payload, self.cfg = self._load_checkpoint(self.checkpoint_path)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = self._create_policy(self.cfg, self.ckpt_payload, self.device)
        self.obs_pose_repr = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr

        if self.verbose:
            print(
                "[policy] initialized | "
                f"device={self.device} "
                f"obs_pose_repr={self.obs_pose_repr} "
                f"action_pose_repr={self.action_pose_repr}"
            )

    def reset(self) -> None:
        self.policy.reset()
        if self.verbose:
            print("[policy] state reset")

    def infer(self, payload: Dict, arm: Optional[str] = None) -> Dict:
        selected_arm, arm_payload = self._select_arm_payload(payload, arm or self.preferred_arm)

        env_obs, episode_start_pose = self._build_env_obs(
            arm_name=selected_arm,
            arm_payload=arm_payload,
            shape_meta=self.cfg.task.shape_meta,
            img_save=self.img_save,
        )
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.cfg.task.shape_meta,
            obs_pose_repr=self.obs_pose_repr,
            tx_robot1_robot0=np.eye(4, dtype=np.float32),
            episode_start_pose=episode_start_pose,
        )
        obs_dict = dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            start_time = time.time()
            result = self.policy.predict_action(obs_dict)
            end_time = time.time()

        raw_action = result["action_pred"][0].detach().cpu().numpy()
        env_action = get_real_umi_action(raw_action, env_obs, self.action_pose_repr)
        action_xyz_quat_grip = self._env_action_to_xyz_quat_gripper(env_action)
        if self.print:
            self._save_policy_output(
                raw_action=raw_action,
                env_action=env_action,
                action_xyz_quat_grip=action_xyz_quat_grip,
                arm_name=selected_arm,
            )

        response = {
            "type": "action",
            "action_l": action_xyz_quat_grip if selected_arm == "arm_l" else [],
            "action_r": action_xyz_quat_grip if selected_arm == "arm_r" else [],
            "timestamp": time.time(),
        }
        if self.verbose:
            print(
                "[policy] inference done | "
                f"selected_arm={selected_arm} "
                f"inference_time={end_time - start_time:.6f}s "
                f"raw_action_shape={list(raw_action.shape)} "
                f"env_action_shape={list(env_action.shape)}"
            )
        return response

    def _load_checkpoint(self, ckpt_path: Path) -> Tuple[dict, object]:
        resolved_ckpt_path = ckpt_path
        if ckpt_path.is_dir():
            resolved_ckpt_path = ckpt_path / "latest.ckpt"
        payload = torch.load(
            open(resolved_ckpt_path, "rb"),
            map_location="cpu",
            pickle_module=dill,
        )
        return payload, payload["cfg"]

    def _create_policy(self, cfg, payload: dict, device: torch.device):
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

    # def _infer_dt(self, cfg) -> float:
    #     dataset_frequency = float(getattr(cfg.task, "dataset_frequeny", 0.0))
    #     obs_down_sample_steps = float(getattr(cfg.task, "obs_down_sample_steps", 1.0))
    #     if dataset_frequency > 0:
    #         return obs_down_sample_steps / dataset_frequency
    #     if self.verbose:
    #         print("[policy] warning: dataset_frequeny <= 0, fallback dt=0.1s")
    #     return 0.1

    def _select_arm_payload(self, payload: dict, arm: str) -> Tuple[str, dict]:
        if "images" in payload and "poses" in payload and "grippers" in payload:
            return arm, payload

        available_arms = [key for key in ("arm_l", "arm_r") if isinstance(payload.get(key), dict)]
        if not available_arms:
            raise ValueError(
                "Payload format is not recognized. Expected flat keys "
                "(`images`, `poses`, `grippers`) or nested `arm_l` / `arm_r`."
            )
        if len(available_arms) == 1:
            selected_arm = available_arms[0]
            if self.verbose:
                print(f"[policy] only one arm found, automatically using `{selected_arm}`")
            return selected_arm, payload[selected_arm]
        if arm not in payload:
            raise ValueError(
                f"Requested arm `{arm}` not found in payload. Available arms: {available_arms}"
            )
        return arm, payload[arm]

    def _build_env_obs(self, arm_name: str, arm_payload: dict, shape_meta: dict, img_save: bool = False) -> Tuple[dict, list]:
        obs_shape_meta = shape_meta["obs"]
        env_obs = {}

        decoded_images = [
            self._decode_rgb_image(image_b64) for image_b64 in arm_payload.get("images", [])
        ]
        if img_save:
            self._save_decoded_images(decoded_images, arm_name)


        poses = [
            self._pose7_to_pos_axis_angle(np.asarray(pose))
            for pose in arm_payload.get("poses", [])
        ]
        grippers = [float(x) for x in arm_payload.get("grippers", [])]

        image_keys = sorted(
            [key for key, attr in obs_shape_meta.items() if attr.get("type", "low_dim") == "rgb"]
        )
        if len(decoded_images) < len(image_keys):
            raise ValueError(
                f"Payload only has {len(decoded_images)} images, but shape_meta requires at least {len(image_keys)}."
            )
        if len(poses) == 0:
            raise ValueError("Payload does not contain any pose history.")
        if len(grippers) == 0:
            raise ValueError("Payload does not contain any gripper history.")

        for key in image_keys:
            horizon = int(obs_shape_meta[key]["horizon"])
            if len(decoded_images) < horizon:
                raise ValueError(
                    f"Payload only has {len(decoded_images)} image frames, but {key} requires horizon={horizon}."
                )
            channels, height, width = obs_shape_meta[key]["shape"]
            if channels != 3:
                raise ValueError(
                    f"Expected 3-channel RGB image for {key}, got shape_meta={obs_shape_meta[key]['shape']}"
                )
            image_hist = decoded_images[-horizon:]
            image_shapes = [img.shape for img in image_hist]
            if len(set(image_shapes)) != 1:
                raise ValueError(f"Inconsistent image shapes in history for {key}: {image_shapes}")
            processed_images = [
                self._crop_resize_rgb_image(
                    image_rgb=img,
                    output_size=(width, height),
                    crop=self.crop,
                )
                for img in image_hist
            ]
            if img_save:
                self._save_processed_images(processed_images, arm_name=arm_name, image_key=key)
            
            env_obs[key] = np.stack(processed_images, axis=0)

        init_pose_raw = arm_payload.get("init_pose")
        if init_pose_raw is None:
            init_pose = poses[0].copy()
        else:
            init_pose = self._pose7_to_pos_axis_angle(np.asarray(init_pose_raw))

        episode_start_pose = [init_pose]
        pos_key = "robot0_eef_pos"
        rot_key = "robot0_eef_rot_axis_angle"
        grip_key = "robot0_gripper_width"

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

    def _save_decoded_images(self, decoded_images: list, arm_name: str) -> None:
        if not decoded_images:
            return
        save_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_input_images"
            / time.strftime("%Y%m%d")
            / f"{arm_name}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for frame_idx, image_rgb in enumerate(decoded_images):
            image_path = save_dir / f"image_{frame_idx:03d}.png"
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(image_path), image_bgr)
            if not success:
                raise ValueError(f"Failed to save decoded image to {image_path}")
            saved_paths.append(str(image_path))
        # if self.verbose:
        #     print(f"[policy] saved decoded images for {arm_name}: {saved_paths}")

    def _save_processed_images(self, processed_images: list, arm_name: str, image_key: str) -> None:
        if not processed_images:
            return
        save_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_processed_images"
            / time.strftime("%Y%m%d")
            / f"{arm_name}_{image_key}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for frame_idx, image_rgb in enumerate(processed_images):
            image_path = save_dir / f"image_{frame_idx:03d}.png"
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(image_path), image_bgr)
            if not success:
                raise ValueError(f"Failed to save processed image to {image_path}")
            saved_paths.append(str(image_path))
        if self.verbose:
            print(f"[policy] saved processed images for {arm_name}/{image_key}: {saved_paths}")

    def _save_policy_output(
        self,
        raw_action: np.ndarray,
        env_action: np.ndarray,
        action_xyz_quat_grip: list,
        arm_name: str,
    ) -> None:
        date_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_output"
            / time.strftime("%Y%m%d")
        )
        date_dir.mkdir(parents=True, exist_ok=True)
        file_path = date_dir / f"{arm_name}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.json"
        file_path.write_text(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "arm_name": arm_name,
                    "raw_action": raw_action.tolist(),
                    "env_action": env_action.tolist(),
                    "action_xyz_quat_grip": action_xyz_quat_grip,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # if self.verbose:
        #     print(f"[policy] saved policy output for {arm_name}: {file_path}")

    def _env_action_to_xyz_quat_gripper(self, env_action: np.ndarray):
        actions = []
        for step in env_action:
            pos = step[:3]
            rotvec = step[3:6]
            grip = step[6]
            quat_xyzw = st.Rotation.from_rotvec(rotvec).as_quat()
            actions.append(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                    float(quat_xyzw[3]),
                    float(grip),
                ]
            )
        return actions

    @staticmethod
    def _decode_rgb_image(image_b64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_b64, validate=True)
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        rgb_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if rgb_image is None:
            raise ValueError("Failed to decode base64 JPEG image.")
        return rgb_image

    @staticmethod
    def _crop_resize_rgb_image(
        image_rgb: np.ndarray,
        output_size: Tuple[int, int],
        crop: bool,
    ) -> np.ndarray:

        if not crop:
            resized = cv2.resize(
                image_rgb, output_size, interpolation=cv2.INTER_AREA
            )
            return resized.astype(np.uint8)
            
        height, width = image_rgb.shape[:2]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        cropped = image_rgb[top:top + crop_size, left:left + crop_size]
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    @staticmethod
    def _pose7_to_pos_axis_angle(pose7: np.ndarray) -> np.ndarray:
        pose7 = np.asarray(pose7, dtype=np.float32)
        if pose7.shape != (7,):
            raise ValueError(f"Expected pose shape (7,), got {pose7.shape}")
        pos = pose7[:3]
        quat_xyzw = pose7[3:]
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
        return np.concatenate([pos, rotvec], axis=0)

#!/usr/bin/env python3
import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation


ARM_STYLES = {
    "arm_l": {
        "action": "#F4D35E",
        "obs": "#1D4ED8",
        "obs_arrow": "#60A5FA",
        "arrow": "#D4A017",
        "init": "#F28C28",
        "image_tag": "#9C2F2F",
    },
    "arm_r": {
        "action": "#F4D35E",
        "obs": "#1D4ED8",
        "obs_arrow": "#60A5FA",
        "arrow": "#D4A017",
        "init": "#F28C28",
        "image_tag": "#3D348B",
    },
}

ACTION_START_COLOR = "#D62828"
ACTION_END_COLOR = "#2A9D8F"
ACTION_GRIPPER_CMAP = cm.plasma

VIEW_SPECS = (
    ("Perspective", 35.0, -55.0),
    ("Top", 90.0, -90.0),
    ("Front", 0.0, -90.0),
    ("Side", 0.0, 0.0),
)

@dataclass
class ArmFrameData:
    arm_name: str
    actions: np.ndarray
    obs_poses: np.ndarray
    init_pose: Optional[np.ndarray]
    latest_image: Optional[np.ndarray]


@dataclass
class FrameData:
    source_path: Path
    timestamp: Optional[float]
    arm_frames: List[ArmFrameData]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize action/observation payloads as multi-view 3D GIF."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single action JSON file or a directory containing action JSON files.",
    )
    parser.add_argument(
        "--output",
        help="Output GIF path. Defaults to <input>/actions_with_obs.gif for directories.",
    )
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate.")
    parser.add_argument("--width", type=float, default=16.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=9.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    parser.add_argument(
        "--arrow-axis",
        choices=("x", "y", "z", "-x", "-y", "-z"),
        default="-z",
        help="Local axis rotated by quaternion and drawn as the action arrow.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion storage order in pose/action arrays.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.18,
        help="Arrow length as a fraction of the global scene span.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate PNG frames in a sibling directory for debugging.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Accepted for compatibility. Rendering is file-oriented by default.",
    )
    return parser.parse_args()


def resolve_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            path for path in input_path.glob("*.json") if path.name.startswith("action_")
        )
        if not files:
            raise FileNotFoundError(f"No action JSON files found in directory: {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_output_path(input_path: Path, output_arg: Optional[str]) -> Path:
    if output_arg:
        output_path = Path(output_arg)
    elif input_path.is_dir():
        output_path = input_path / "actions_with_obs.gif"
    else:
        output_path = input_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def decode_image_from_b64(image_b64: str) -> Optional[np.ndarray]:
    if not isinstance(image_b64, str) or not image_b64:
        return None
    image_bytes = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.array(image.convert("RGB"))


def ensure_pose_array(values: Sequence[Sequence[float]], min_width: int = 7) -> np.ndarray:
    if not isinstance(values, Sequence) or len(values) == 0:
        return np.zeros((0, min_width), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < min_width:
        return np.zeros((0, min_width), dtype=np.float32)
    return array


def ensure_pose_vector(values: Optional[Sequence[float]], min_width: int = 7) -> Optional[np.ndarray]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] < min_width:
        return None
    return array


def load_frame_data(json_path: Path) -> FrameData:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    observation = payload.get("observation", {})
    arm_frames: List[ArmFrameData] = []

    for arm_name in ("arm_l", "arm_r"):
        action_key = "action_l" if arm_name == "arm_l" else "action_r"
        actions = ensure_pose_array(payload.get(action_key, []), min_width=7)
        arm_obs = observation.get(arm_name, {}) if isinstance(observation, dict) else {}
        obs_poses = ensure_pose_array(arm_obs.get("poses", []), min_width=7)
        init_pose = ensure_pose_vector(arm_obs.get("init_pose"), min_width=7)
        latest_image = None

        images = arm_obs.get("images", [])
        if isinstance(images, list) and images:
            latest_image = decode_image_from_b64(images[-1])

        if (
            actions.shape[0] == 0
            and obs_poses.shape[0] == 0
            and init_pose is None
            and latest_image is None
        ):
            continue

        arm_frames.append(
            ArmFrameData(
                arm_name=arm_name,
                actions=actions,
                obs_poses=obs_poses,
                init_pose=init_pose,
                latest_image=latest_image,
            )
        )

    return FrameData(
        source_path=json_path,
        timestamp=payload.get("timestamp"),
        arm_frames=arm_frames,
    )


def gather_scene_bounds(frames: Sequence[FrameData]) -> Tuple[np.ndarray, float]:
    points: List[np.ndarray] = []
    for frame in frames:
        for arm in frame.arm_frames:
            if arm.actions.size:
                points.append(arm.actions[:, :3])
            if arm.obs_poses.size:
                points.append(arm.obs_poses[:, :3])
            if arm.init_pose is not None:
                points.append(arm.init_pose[None, :3])

    if not points:
        return np.zeros(3, dtype=np.float32), 1.0

    stacked = np.concatenate(points, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    return center.astype(np.float32), max(span, 1e-3)


def get_action_gripper_widths(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if actions.shape[1] >= 8:
        return actions[:, -1].astype(np.float32)
    return np.zeros((actions.shape[0],), dtype=np.float32)


def gather_gripper_width_range(frames: Sequence[FrameData]) -> Tuple[float, float]:
    values: List[np.ndarray] = []
    for frame in frames:
        for arm in frame.arm_frames:
            widths = get_action_gripper_widths(arm.actions)
            if widths.size:
                values.append(widths)
    if not values:
        return 0.0, 1.0
    stacked = np.concatenate(values, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-6
    return vmin, vmax


def axis_vector(axis_name: str) -> np.ndarray:
    if axis_name == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if axis_name == "-x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    if axis_name == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if axis_name == "-y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if axis_name == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, -1.0], dtype=np.float32)


def as_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat_order == "xyzw":
        return quat
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def set_axes_equal(ax: plt.Axes, scene_center: np.ndarray, scene_span: float) -> None:
    half = scene_span * 0.6
    center = scene_center.astype(np.float32)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def style_3d_axis(ax: plt.Axes, title: str, scene_center: np.ndarray, scene_span: float) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax, scene_center, scene_span)
    ax.grid(True, alpha=0.35)


def draw_world_axes(ax: plt.Axes, scene_center: np.ndarray, scene_span: float) -> None:
    axis_len = scene_span * 0.18
    origin = scene_center
    axes = (
        (np.array([axis_len, 0.0, 0.0]), "r", "X"),
        (np.array([0.0, axis_len, 0.0]), "g", "Y"),
        (np.array([0.0, 0.0, axis_len]), "b", "Z"),
    )
    for delta, color, label in axes:
        end = origin + delta
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            delta[0],
            delta[1],
            delta[2],
            color=color,
            linewidth=1.4,
            arrow_length_ratio=0.18,
        )
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=9)


def plot_arm_data(
    ax: plt.Axes,
    arm: ArmFrameData,
    quat_order: str,
    arrow_axis_name: str,
    arrow_len: float,
    grip_norm: mcolors.Normalize,
) -> None:
    style = ARM_STYLES.get(
        arm.arm_name,
        {
            "action": "#666666",
            "obs": "#AAAAAA",
            "obs_arrow": "#BBBBBB",
            "arrow": "#444444",
            "init": "#000000",
            "image_tag": "#333333",
        },
    )
    obs_color = style["obs"]
    obs_arrow_color = style["obs_arrow"]
    init_color = style["init"]

    if arm.obs_poses.size:
        obs_xyz = arm.obs_poses[:, :3]
        ax.scatter(
            obs_xyz[:, 0],
            obs_xyz[:, 1],
            obs_xyz[:, 2],
            color=obs_color,
            edgecolors="white",
            linewidths=0.9,
            s=160,
            alpha=1.0,
            label=f"{arm.arm_name} obs poses",
        )
        for obs_pose in arm.obs_poses:
            obs_quat_xyzw = as_xyzw(obs_pose[3:7], quat_order)
            obs_direction = Rotation.from_quat(obs_quat_xyzw).apply(axis_vector(arrow_axis_name))
            obs_delta = obs_direction * arrow_len
            ax.quiver(
                obs_pose[0],
                obs_pose[1],
                obs_pose[2],
                obs_delta[0],
                obs_delta[1],
                obs_delta[2],
                color=obs_arrow_color,
                linewidth=1.8,
                alpha=0.95,
                arrow_length_ratio=0.32,
            )

    if arm.init_pose is not None:
        init_xyz = arm.init_pose[:3]
        ax.scatter(
            [init_xyz[0]],
            [init_xyz[1]],
            [init_xyz[2]],
            color=init_color,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            s=180,
            label=f"{arm.arm_name} init_pose",
        )

    if not arm.actions.size:
        return

    action_xyz = arm.actions[:, :3]
    gripper_widths = get_action_gripper_widths(arm.actions)
    if len(action_xyz) >= 2:
        segments = np.stack([action_xyz[:-1], action_xyz[1:]], axis=1)
        segment_widths = 0.5 * (gripper_widths[:-1] + gripper_widths[1:])
        line_collection = Line3DCollection(
            segments,
            cmap=ACTION_GRIPPER_CMAP,
            norm=grip_norm,
            linewidth=2.8,
            alpha=0.98,
        )
        line_collection.set_array(segment_widths)
        ax.add_collection3d(line_collection)

    ax.scatter(
        action_xyz[:, 0],
        action_xyz[:, 1],
        action_xyz[:, 2],
        c=gripper_widths,
        cmap=ACTION_GRIPPER_CMAP,
        norm=grip_norm,
        edgecolors="black",
        linewidths=0.35,
        s=28,
        label=f"{arm.arm_name} actions (gripper width)",
    )
    start_xyz = action_xyz[0]
    end_xyz = action_xyz[-1]
    ax.scatter(
        [start_xyz[0]],
        [start_xyz[1]],
        [start_xyz[2]],
        color=ACTION_START_COLOR,
        edgecolors="black",
        linewidths=0.6,
        s=92,
        label=f"{arm.arm_name} action start",
        zorder=6,
    )
    ax.scatter(
        [end_xyz[0]],
        [end_xyz[1]],
        [end_xyz[2]],
        color=ACTION_END_COLOR,
        edgecolors="black",
        linewidths=0.6,
        s=92,
        label=f"{arm.arm_name} action end",
        zorder=6,
    )

    axis_local = axis_vector(arrow_axis_name)
    for index, action in enumerate(arm.actions):
        xyz = action[:3]
        quat_xyzw = as_xyzw(action[3:7], quat_order)
        direction = Rotation.from_quat(quat_xyzw).apply(axis_local)
        delta = direction * arrow_len
        arrow_color = ACTION_GRIPPER_CMAP(grip_norm(float(gripper_widths[index])))
        ax.quiver(
            xyz[0],
            xyz[1],
            xyz[2],
            delta[0],
            delta[1],
            delta[2],
            color=arrow_color,
            linewidth=1.3,
            alpha=0.9,
            arrow_length_ratio=0.28,
        )


def draw_pose_view(
    ax: plt.Axes,
    frame: FrameData,
    title: str,
    elev: float,
    azim: float,
    scene_center: np.ndarray,
    scene_span: float,
    quat_order: str,
    arrow_axis_name: str,
    arrow_scale: float,
    grip_norm: mcolors.Normalize,
) -> None:
    style_3d_axis(ax, title, scene_center, scene_span)
    draw_world_axes(ax, scene_center, scene_span)
    arrow_len = scene_span * arrow_scale

    for arm in frame.arm_frames:
        plot_arm_data(
            ax=ax,
            arm=arm,
            quat_order=quat_order,
            arrow_axis_name=arrow_axis_name,
            arrow_len=arrow_len,
            grip_norm=grip_norm,
        )

    ax.view_init(elev=elev, azim=azim)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)


def draw_image_panel(ax: plt.Axes, frame: FrameData) -> None:
    ax.set_title("Latest Observation Image", fontsize=11)
    ax.axis("off")

    latest_image = None
    image_arm = None
    for arm in frame.arm_frames:
        if arm.latest_image is not None:
            latest_image = arm.latest_image
            image_arm = arm.arm_name
            break

    if latest_image is None:
        ax.text(0.5, 0.5, "No observation image", ha="center", va="center", fontsize=12)
        return

    ax.imshow(latest_image)
    if image_arm is not None:
        style = ARM_STYLES.get(image_arm, {"image_tag": "#333333"})
        ax.text(
            0.02,
            0.02,
            image_arm,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            bbox={"facecolor": style["image_tag"], "alpha": 0.8, "pad": 4},
        )


def draw_info_panel(ax: plt.Axes, frame: FrameData, grip_range: Tuple[float, float]) -> None:
    ax.set_title("Frame Info", fontsize=11)
    ax.axis("off")
    lines = [
        f"file: {frame.source_path.name}",
        f"timestamp: {frame.timestamp:.6f}" if frame.timestamp is not None else "timestamp: n/a",
        "",
        "Legend",
        "- action color: gripper width",
        "- red point: action start",
        "- green point: action end",
        "- arrows use same gripper colormap",
        "- large blue points: observation poses",
        "- light blue arrows: observation orientation",
        "- star: init_pose",
        "",
        f"gripper width range: [{grip_range[0]:.4f}, {grip_range[1]:.4f}]",
        "",
    ]

    for arm in frame.arm_frames:
        lines.append(
            f"{arm.arm_name}: actions={len(arm.actions)} obs={len(arm.obs_poses)} image={'yes' if arm.latest_image is not None else 'no'}"
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )


def render_frame(
    frame: FrameData,
    scene_center: np.ndarray,
    scene_span: float,
    grip_range: Tuple[float, float],
    width: float,
    height: float,
    dpi: int,
    quat_order: str,
    arrow_axis_name: str,
    arrow_scale: float,
) -> Image.Image:
    fig = plt.figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    grip_norm = mcolors.Normalize(vmin=grip_range[0], vmax=grip_range[1])

    axes_3d = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
        fig.add_subplot(gs[0, 2], projection="3d"),
        fig.add_subplot(gs[1, 0], projection="3d"),
    ]
    image_ax = fig.add_subplot(gs[1, 1])
    info_ax = fig.add_subplot(gs[1, 2])

    for ax, (title, elev, azim) in zip(axes_3d, VIEW_SPECS):
        draw_pose_view(
            ax=ax,
            frame=frame,
            title=title,
            elev=elev,
            azim=azim,
            scene_center=scene_center,
            scene_span=scene_span,
            quat_order=quat_order,
            arrow_axis_name=arrow_axis_name,
            arrow_scale=arrow_scale,
            grip_norm=grip_norm,
        )

    draw_image_panel(image_ax, frame)
    draw_info_panel(info_ax, frame, grip_range)
    colorbar_mappable = cm.ScalarMappable(norm=grip_norm, cmap=ACTION_GRIPPER_CMAP)
    colorbar_mappable.set_array([])
    colorbar = fig.colorbar(
        colorbar_mappable,
        ax=axes_3d,
        fraction=0.03,
        pad=0.03,
        shrink=0.9,
    )
    colorbar.set_label("Action gripper width", fontsize=10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as pil_image:
        return pil_image.convert("RGB")


def render_all_frames(
    frames: Sequence[FrameData],
    output_path: Path,
    fps: int,
    width: float,
    height: float,
    dpi: int,
    quat_order: str,
    arrow_axis_name: str,
    arrow_scale: float,
    keep_frames: bool,
) -> None:
    scene_center, scene_span = gather_scene_bounds(frames)
    grip_range = gather_gripper_width_range(frames)
    rendered_frames: List[Image.Image] = []
    frame_dir: Optional[Path] = None

    if keep_frames:
        frame_dir = output_path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        image = render_frame(
            frame=frame,
            scene_center=scene_center,
            scene_span=scene_span,
            grip_range=grip_range,
            width=width,
            height=height,
            dpi=dpi,
            quat_order=quat_order,
            arrow_axis_name=arrow_axis_name,
            arrow_scale=arrow_scale,
        )
        rendered_frames.append(image)
        if frame_dir is not None:
            image.save(frame_dir / f"frame_{index:05d}.png")

    if not rendered_frames:
        raise RuntimeError("No rendered frames were produced.")

    duration_ms = max(1, int(round(1000 / max(fps, 1))))
    rendered_frames[0].save(
        output_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = resolve_output_path(input_path, args.output)
    input_files = resolve_input_files(input_path)
    frames = [load_frame_data(path) for path in input_files]

    if not frames:
        raise RuntimeError("No frames were loaded from the input.")

    render_all_frames(
        frames=frames,
        output_path=output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        quat_order=args.quat_order,
        arrow_axis_name=args.arrow_axis,
        arrow_scale=args.arrow_scale,
        keep_frames=args.keep_frames,
    )
    print(f"Saved GIF to: {output_path}")


if __name__ == "__main__":
    main()

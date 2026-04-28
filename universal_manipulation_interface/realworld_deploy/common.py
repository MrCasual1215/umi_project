import base64
import json
import socket
import time
from pathlib import Path
from typing import Dict, Tuple

def validate_pose_6d(pose_6d):
    if not isinstance(pose_6d, (list, tuple)) or len(pose_6d) != 6:
        raise ValueError(f"pose_6d must be a list/tuple of length 6, got: {pose_6d}")
    return [float(x) for x in pose_6d]


def send_json_line(sock: socket.socket, payload: Dict, encoding: str = "utf-8") -> None:
    message = json.dumps(payload, separators=(",", ":")) + "\n"
    sock.sendall(message.encode(encoding))


def recv_json_line(
    sock: socket.socket,
    recv_buffer: bytes = b"",
    encoding: str = "utf-8",
    buffer_size: int = 4096,
) -> Tuple[Dict, bytes]:
    chunks = recv_buffer
    while True:
        while b"\n" not in chunks:
            part = sock.recv(buffer_size)
            if not part:
                raise ConnectionError("Socket closed while waiting for a newline-delimited JSON message.")
            chunks += part
        line, chunks = chunks.split(b"\n", 1)
        if not line.strip():
            continue
        return json.loads(line.decode(encoding)), chunks


def build_status_response(message_type: str, **extra) -> Dict:
    payload = {
        "type": message_type,
        "timestamp": time.time(),
    }
    payload.update(extra)
    return payload


def decode_jpeg_to_png(image_b64: str) -> bytes:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Converting images to PNG requires `opencv-python` and `numpy` to be installed."
        ) from exc

    jpeg_bytes = base64.b64decode(image_b64, validate=True)
    image_buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    rgb_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR) 
    if rgb_image is None:
        raise ValueError("Failed to decode JPG image data.")
    # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    _, _, channels = rgb_image.shape
    if channels != 3:
        raise ValueError(f"Expected 3 RGB channels, got {channels}")

    png_ready_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    success, encoded_png = cv2.imencode(".png", png_ready_bgr)
    if not success:
        raise ValueError("Failed to encode PNG image data.")
    return encoded_png.tobytes()


def iter_payload_images(payload: Dict):
    images = payload.get("images")
    if isinstance(images, list):
        for index, image_b64 in enumerate(images):
            yield f"image_{index:03d}.png", image_b64

    for arm_name in ("arm_l", "arm_r"):
        arm_payload = payload.get(arm_name)
        if not isinstance(arm_payload, dict):
            continue
        arm_images = arm_payload.get("images")
        if not isinstance(arm_images, list):
            continue
        for index, image_b64 in enumerate(arm_images):
            yield f"{arm_name}_image_{index:03d}.png", image_b64


def save_payload_record(payload: Dict, save_root: Path, pict_save: bool) -> Dict:
    date_dir = save_root / time.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    message_id = f"obs_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    save_dir = date_dir / message_id
    save_dir.mkdir(parents=True, exist_ok=False)

    (save_dir / "payload_raw.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (save_dir / "receive_info.json").write_text(
        json.dumps(
            {
                "received_timestamp": time.time(),
                "payload_type": payload.get("type"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    saved_images = []
    if pict_save:
        for image_name, image_b64 in iter_payload_images(payload):
            if not isinstance(image_b64, str):
                continue
            (save_dir / image_name).write_bytes(decode_jpeg_to_png(image_b64))
            saved_images.append(image_name)

    return {
        "save_dir": str(save_dir),
        "payload_type": payload.get("type"),
        "saved_images": saved_images,
    }


def save_response_record(response: Dict, save_root: Path) -> Dict:
    date_dir = save_root / time.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    response_type = str(response.get("type", "response"))
    file_path = date_dir / f"{response_type}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.json"
    file_path.write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "file_path": str(file_path),
        "response_type": response_type,
    }


def rgb_bytes_to_message(rgb_bytes: bytes, width: int, height: int) -> Dict:
    if len(rgb_bytes) != width * height * 3:
        raise ValueError("RGB byte length does not match width * height * 3.")
    return {
        "encoding": "raw_rgb_u8",
        "width": int(width),
        "height": int(height),
        "channels": 3,
        "data_b64": base64.b64encode(rgb_bytes).decode("ascii"),
    }


def message_to_rgb_bytes(rgb_message: Dict) -> Tuple[bytes, int, int]:
    width = int(rgb_message["width"])
    height = int(rgb_message["height"])
    channels = int(rgb_message["channels"])
    if channels != 3:
        raise ValueError(f"Only 3-channel RGB is supported, got channels={channels}")
    rgb_bytes = base64.b64decode(rgb_message["data_b64"])
    expected_len = width * height * channels
    if len(rgb_bytes) != expected_len:
        raise ValueError(
            f"RGB payload size mismatch: expected {expected_len} bytes, got {len(rgb_bytes)} bytes"
        )
    return rgb_bytes, width, height

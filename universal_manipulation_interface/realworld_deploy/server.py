#!/usr/bin/env python3
import socket
import time
from pathlib import Path
from typing import Dict, Optional

from common import (
    build_status_response,
    recv_json_line,
    save_payload_record,
    save_response_record,
    send_json_line,
)
from policy_inference import PolicyInference
from config import (
    SERVER_HOST,
    SERVER_PORT,
    SOCKET_TIMEOUT_SEC,
    BUFFER_SIZE,
    ENCODING,
    MAX_CLIENTS,
    POLICY_CHECKPOINT_PATH,
    DEFAULT_POLICY_ARM,
    VERBOSE,
    PRINT,
    PICT_SAVE,
    CROP,
    ACTION_CHUNK_HORIZON,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
SAVE_ROOT = OUTPUT_ROOT / "received_observations"
RAW_ACTION_ROOT = OUTPUT_ROOT / "raw_actions"
SENT_ROOT = OUTPUT_ROOT / "sent_actions"
RAW_RECEIVED_JSON_ROOT = OUTPUT_ROOT / "raw_received_json"
RAW_SENT_JSON_ROOT = OUTPUT_ROOT / "raw_sent_json"
EMPTY_SAVE_INFO = {"save_dir": None, "payload_type": None, "saved_images": []}
POLICY = PolicyInference(
    checkpoint_path=POLICY_CHECKPOINT_PATH,
    preferred_arm=DEFAULT_POLICY_ARM,
    verbose=VERBOSE,
    _print=PRINT,
    img_save=PICT_SAVE,
    crop=CROP,
)
LATEST_OBSERVATION_PAYLOAD = None
LAST_ACTION_RESPONSE = None
LAST_INFERENCE_MONOTONIC = None


def reset_handler() -> None:
    global LATEST_OBSERVATION_PAYLOAD, LAST_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC
    POLICY.reset()
    LATEST_OBSERVATION_PAYLOAD = None
    LAST_ACTION_RESPONSE = None
    LAST_INFERENCE_MONOTONIC = None


def maybe_save_payload(payload: Dict) -> Dict:
    if not PRINT:
        return {
            **EMPTY_SAVE_INFO,
            "payload_type": payload.get("type"),
        }
    return save_payload_record(payload, SAVE_ROOT, PICT_SAVE)


def maybe_save_response(response: Dict) -> Dict:
    if not PRINT:
        return {
            "file_path": None,
            "response_type": response.get("type"),
        }
    return save_response_record(response, SENT_ROOT)


def maybe_save_raw_action(response: Dict) -> Dict:
    if not PRINT:
        return {
            "file_path": None,
            "response_type": response.get("type"),
        }
    return save_response_record(response, RAW_ACTION_ROOT)


def save_raw_received_json(payload: Dict) -> Dict:
    return save_response_record(payload, RAW_RECEIVED_JSON_ROOT)


def save_raw_sent_json(response: Dict) -> Dict:
    return save_response_record(response, RAW_SENT_JSON_ROOT)


def extract_observation_arm_payload(payload: Dict) -> Dict:
    if "arm_l" in payload or "arm_r" in payload:
        filtered_payload = {}
        for arm_name in ("arm_l", "arm_r"):
            arm_payload = payload.get(arm_name)
            if isinstance(arm_payload, dict):
                filtered_payload[arm_name] = arm_payload
        return filtered_payload
    return payload


def build_saved_inference_record(observation_payload: Dict, action_response: Dict) -> Dict:
    record = dict(action_response)
    record["observation"] = extract_observation_arm_payload(observation_payload)
    return record


def truncate_action_chunk(response: Dict) -> Dict:
    horizon = ACTION_CHUNK_HORIZON
    if horizon is None:
        return response
    if horizon < 0:
        raise ValueError("ACTION_CHUNK_HORIZON must be greater than or equal to 0.")

    truncated_response = dict(response)
    for action_key in ("action_l", "action_r"):
        action_chunk = truncated_response.get(action_key)
        if isinstance(action_chunk, list):
            truncated_response[action_key] = action_chunk[:horizon]
    return truncated_response


def build_shakehands_response(payload: Dict, received_timestamp: float) -> Dict:
    arm = DEFAULT_POLICY_ARM
    for arm_name in ("arm_l", "arm_r"):
        if isinstance(payload.get(arm_name), dict):
            arm = arm_name
            break
    return {
        "type": "shakehands",
        "arm": arm,
        "sent_timestamp": time.time(),
        "received_timestamp": received_timestamp,
    }


def prepare_observation(payload: Dict) -> Dict:
    global LATEST_OBSERVATION_PAYLOAD, LAST_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC
    LATEST_OBSERVATION_PAYLOAD = payload
    received_timestamp = time.time()
    arm_count = sum(
        1
        for arm_name in ("arm_l", "arm_r")
        if isinstance(payload.get(arm_name), dict)
    )

    if VERBOSE:
        message = (
            "[server] observation received | "
            f"payload_type={payload.get('type')}"
        )
        if arm_count:
            message += f" arms={arm_count}"
        print(message)

    return build_shakehands_response(
        payload=LATEST_OBSERVATION_PAYLOAD,
        received_timestamp=received_timestamp,
    )


def run_observation_inference() -> Dict:
    global LAST_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC
    maybe_save_payload(LATEST_OBSERVATION_PAYLOAD)
    start_time = time.monotonic()
    LAST_ACTION_RESPONSE = truncate_action_chunk(POLICY.infer(LATEST_OBSERVATION_PAYLOAD))
    end_time = time.monotonic()
    LAST_INFERENCE_MONOTONIC = time.monotonic()
    if VERBOSE:
        message = "[server] policy triggered immediately | "
        message += f"inference_time={end_time - start_time:.6f}s "
        message += f"action_chunk_horizon={ACTION_CHUNK_HORIZON}"
        print(message)
    return LAST_ACTION_RESPONSE


def handle_reset(_payload: Dict):
    reset_handler()
    if VERBOSE:
        print("[server] reset received")
    return build_status_response("reset_ack"), None, False


def handle_message(payload: Dict) -> tuple[Optional[Dict], Optional[Dict], bool]:
    message_type = payload.get("type")
    if VERBOSE:
        print(f"[server] received message type: {message_type}")
    if message_type == "reset":
        return handle_reset(payload)
    raise ValueError(f"Unsupported message type: {message_type}")

# TODO: 更新时间戳
def serve_forever() -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(MAX_CLIENTS)

    print(f"[server] listening on {SERVER_HOST}:{SERVER_PORT}")

    try:
        while True:
            print("[server] waiting for client connection...")
            client_socket, client_addr = server_socket.accept()
            print(f"[server] client connected: {client_addr}")
            client_socket.settimeout(SOCKET_TIMEOUT_SEC)
            recv_buffer = b""

            try:
                while True:
                    payload, recv_buffer = recv_json_line(
                        client_socket,
                        recv_buffer,
                        encoding=ENCODING,
                        buffer_size=BUFFER_SIZE,
                    )
                    raw_received_save_info = save_raw_received_json(payload)
                    if VERBOSE:
                        print(
                            "[server] raw request saved | "
                            f"type={raw_received_save_info['response_type']} "
                        )
                    if payload.get("type") == "observation":
                        ## shakehands
                        shakehands_response = prepare_observation(payload)
                        raw_sent_save_info = save_raw_sent_json(shakehands_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        # send_json_line(client_socket, shakehands_response, encoding=ENCODING)

                        ## run inference
                        response = run_observation_inference()
                        raw_action_save_info = maybe_save_raw_action(response)
                        if VERBOSE and raw_action_save_info["file_path"] is not None:
                            print(
                                "[server] raw action saved | "
                                f"type={raw_action_save_info['response_type']} "
                                # f"path={raw_action_save_info['file_path']}"
                            )
                        saved_response = build_saved_inference_record(
                            observation_payload=LATEST_OBSERVATION_PAYLOAD,
                            action_response=response,
                        )
                        response_save_info = maybe_save_response(saved_response)
                        if VERBOSE and response_save_info["file_path"] is not None:
                            print(
                                "[server] response saved | "
                                f"type={response_save_info['response_type']} "
                                # f"path={response_save_info['file_path']}"
                            )
                        raw_sent_save_info = save_raw_sent_json(response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        send_json_line(client_socket, response, encoding=ENCODING)
                        continue

                    immediate_response, response, should_save_response = handle_message(payload)
                    if immediate_response is not None:
                        raw_sent_save_info = save_raw_sent_json(immediate_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        send_json_line(client_socket, immediate_response, encoding=ENCODING)
                    if response is None:
                        continue
                    if should_save_response:
                        raw_action_save_info = maybe_save_raw_action(response)
                        if VERBOSE and raw_action_save_info["file_path"] is not None:
                            print(
                                "[server] raw action saved | "
                                f"type={raw_action_save_info['response_type']} "
                                # f"path={raw_action_save_info['file_path']}"
                            )
                        saved_response = build_saved_inference_record(
                            observation_payload=LATEST_OBSERVATION_PAYLOAD,
                            action_response=response,
                        )
                        response_save_info = maybe_save_response(saved_response)
                        if VERBOSE and response_save_info["file_path"] is not None:
                            print(
                                "[server] response saved | "
                                f"type={response_save_info['response_type']} "
                                # f"path={response_save_info['file_path']}"
                            )
                    raw_sent_save_info = save_raw_sent_json(response)
                    if VERBOSE:
                        print(
                            "[server] raw response saved | "
                            f"type={raw_sent_save_info['response_type']} "
                        )
                    send_json_line(client_socket, response, encoding=ENCODING)
            except ConnectionError:
                print(f"[server] client disconnected: {client_addr}")
            except socket.timeout:
                print(f"[server] client timeout: {client_addr}")
            except Exception as exc:
                print(f"[server] error while handling {client_addr}: {exc}")
            finally:
                client_socket.close()
    finally:
        server_socket.close()


if __name__ == "__main__":
    serve_forever()

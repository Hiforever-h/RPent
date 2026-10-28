# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboTwin EEF16 trajectory selection and conversion for Flywheel."""

from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.reader import read_array, validate_image

IMAGE_FIELDS = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
DATASET_FPS = 20


def successful_prefix(actions: list[dict]) -> int:
    """Return the prefix through the first native success, or zero."""
    for index, action in enumerate(actions):
        result = action.get("result") or {}
        if result.get("format") != "robotwin.step_result":
            continue
        outcome = result.get("data") or {}
        if outcome.get("terminated") is True and not any(
            key in result.get("missing", {}) for key in ("", "/terminated")
        ):
            return index + 1
        if outcome.get("truncated") is True:
            return 0
    return 0


def read_actions(root: Path, payload: dict[str, Any]) -> np.ndarray:
    """Read one executed EEF16 target, retaining pose and gripper values."""
    payload = payload or {}
    if (
        payload.get("format") != "robotwin.action.ee"
        or payload.get("completeness") != "complete"
        or payload.get("missing")
    ):
        raise ValueError("Missing or unsupported RoboTwin EEF16 action payload")
    values = read_array(root, payload["data"])
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("RoboTwin actions must contain sixteen finite values")
    return values


def validate_observation(root: Path, observation: dict[str, Any]) -> np.ndarray:
    """Check required observation fields and return the EEF16 state."""
    payload = observation.get("payload") or {}
    if payload.get("format") != "robotwin.observation":
        raise ValueError("Expected a RoboTwin observation")
    required = (
        "",
        "/robot_state",
        "/observation",
        *(
            f"/robot_state/{side}_{name}"
            for side in ("left", "right")
            for name in ("eef_pose", "gripper")
        ),
        *(
            path
            for camera in IMAGE_FIELDS.values()
            for path in (f"/observation/{camera}", f"/observation/{camera}/rgb")
        ),
    )
    if observation.get("missing", {}).get("payload") or any(
        key in payload.get("missing", {}) for key in required
    ):
        raise ValueError("Missing required RoboTwin observation data")
    fields = payload["data"]
    robot = fields["robot_state"]
    parts = []
    for side in ("left", "right"):
        pose = read_array(root, robot[f"{side}_eef_pose"])
        if pose.shape != (7,):
            raise ValueError("RoboTwin end-effector poses must have seven values")
        parts.extend([pose, np.asarray([robot[f"{side}_gripper"]], dtype=pose.dtype)])
    state = np.concatenate(parts)
    if not np.isfinite(state).all():
        raise ValueError("RoboTwin state must contain sixteen finite values")
    for camera in IMAGE_FIELDS.values():
        validate_image(fields["observation"][camera]["rgb"], f"RoboTwin {camera} image")
    return state


def read_frame(
    root: Path, action_record: dict[str, Any], instruction: str
) -> dict[str, Any]:
    """Read one executed EEF16 action and its before observation."""
    observation = action_record["before_observation"]
    state = validate_observation(root, observation)
    fields = observation["payload"]["data"]
    return {
        **{
            name: read_array(root, fields["observation"][camera]["rgb"])
            for name, camera in IMAGE_FIELDS.items()
        },
        "observation.state": state,
        "action": read_actions(root, action_record["action"]),
        "prompt": instruction,
    }

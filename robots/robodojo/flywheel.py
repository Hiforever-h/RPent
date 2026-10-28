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

"""RoboDojo ARX-X5 trajectory selection and conversion for Flywheel."""

from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.reader import read_array, validate_image

IMAGE_FIELDS = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}
JOINT_FIELDS = {
    "left_arm_joint_state": 6,
    "right_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_ee_joint_state": 1,
}
DATASET_FPS = 25


def successful_prefix(actions: list[dict]) -> int:
    """Return the prefix through the first native success, or zero."""
    for index, action in enumerate(actions):
        result = action.get("result") or {}
        if (
            result.get("format") == "robodojo.step_result"
            and (result.get("data") or {}).get("done") is True
            and not any(key in result.get("missing", {}) for key in ("", "/done"))
        ):
            return index + 1
    return 0


def _read_joints(root: Path, fields: dict[str, Any]) -> np.ndarray:
    """Join both arms, then both grippers, in ARX-X5 policy order."""
    parts = []
    for name, size in JOINT_FIELDS.items():
        value = fields[name]
        part = (
            read_array(root, value)
            if isinstance(value, dict)
            else np.asarray(value, dtype=np.float32)
        )
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"RoboDojo {name} must contain {size} finite values")
        parts.append(part)
    return np.concatenate(parts)


def read_actions(root: Path, payload: dict[str, Any]) -> np.ndarray:
    """Read one absolute joint target in ARX-X5 policy order."""
    payload = payload or {}
    if (
        payload.get("format") != "robodojo.action.joint"
        or payload.get("completeness") != "complete"
        or payload.get("missing")
    ):
        raise ValueError("Missing or unsupported RoboDojo joint action payload")
    return _read_joints(root, payload["data"])


def validate_observation(root: Path, observation: dict[str, Any]) -> np.ndarray:
    """Check required observation fields and return the 14-value joint state."""
    payload = observation.get("payload") or {}
    if payload.get("format") != "robodojo.observation":
        raise ValueError("Expected a RoboDojo observation")
    required = (
        "",
        "/state",
        "/vision",
        *(f"/state/{name}" for name in JOINT_FIELDS),
        *(
            path
            for camera in IMAGE_FIELDS.values()
            for path in (f"/vision/{camera}", f"/vision/{camera}/color")
        ),
    )
    if observation.get("missing", {}).get("payload") or any(
        key in payload.get("missing", {}) for key in required
    ):
        raise ValueError("Missing required RoboDojo observation data")
    fields = payload["data"]
    state = _read_joints(root, fields["state"])
    for camera in IMAGE_FIELDS.values():
        validate_image(fields["vision"][camera]["color"], f"RoboDojo {camera} image")
    return state


def read_frame(
    root: Path, action_record: dict[str, Any], instruction: str
) -> dict[str, Any]:
    """Read one executed joint action and its before observation."""
    observation = action_record["before_observation"]
    state = validate_observation(root, observation)
    fields = observation["payload"]["data"]
    return {
        **{
            name: read_array(root, fields["vision"][camera]["color"])
            for name, camera in IMAGE_FIELDS.items()
        },
        "observation.state": state,
        "action": read_actions(root, action_record["action"]),
        "prompt": instruction,
    }

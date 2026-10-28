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

"""LIBERO trajectory selection and conversion for Flywheel."""

from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.reader import read_array, validate_image

IMAGE_FIELDS = {
    "image": "main_images",
    "wrist_image": "wrist_images",
}
DATASET_FPS = 20


def successful_prefix(actions: list[dict]) -> int:
    """Return the prefix length through the first environment success, or zero."""
    for index, action in enumerate(actions):
        result = action.get("result") or {}
        if result.get("format") != "libero.step_result":
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
    """Read a native action or executed prediction chunk in the same action space."""
    dimensions = {"libero.action": 1, "libero.action_chunk": 2}
    payload = payload or {}
    ndim = dimensions.get(payload.get("format"))
    if (
        ndim is None
        or payload.get("completeness") != "complete"
        or payload.get("missing")
    ):
        raise ValueError("Missing or unsupported LIBERO action payload")
    values = read_array(root, payload.get("data") or {})
    if values.ndim != ndim or values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError("LIBERO actions must contain seven finite values per step")
    return values


def read_frame(
    root: Path, action_record: dict[str, Any], instruction: str
) -> dict[str, Any]:
    """Read one executed action and its before observation as a training frame."""
    observation = action_record["before_observation"]
    state = validate_observation(root, observation)
    actions = read_actions(root, action_record["action"])
    if actions.ndim != 1:
        raise ValueError("Expected one executed action")
    fields = observation["payload"]["data"]
    return {
        **{
            name: read_array(root, fields[field])
            for name, field in IMAGE_FIELDS.items()
        },
        "state": state,
        "actions": actions,
        "prompt": instruction,
    }


def validate_observation(root: Path, observation: dict[str, Any]) -> np.ndarray:
    """Check required observation fields and return the state, leaving images unloaded."""
    payload = observation.get("payload") or {}
    fields = payload.get("data") or {}
    if payload.get("format") != "libero.observation":
        raise ValueError("Expected a LIBERO observation")
    if observation.get("missing", {}).get("payload") or any(
        key in payload.get("missing", {})
        for key in ("", "/states", *(f"/{name}" for name in IMAGE_FIELDS.values()))
    ):
        raise ValueError("Missing required LIBERO observation data")
    state = read_array(root, fields.get("states") or {})
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO state must contain eight finite values")
    for name, field in IMAGE_FIELDS.items():
        validate_image(
            fields.get(field), f"LIBERO {name}", expected_shape=(256, 256, 3)
        )
    return state

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

"""LIBERO success and native data checks for offline trajectory selection."""

from pathlib import Path
from typing import Any

import numpy as np

from rpent.data.reader import read_array

OBSERVATION_FIELDS = {
    "image": "main_images",
    "wrist_image": "wrist_images",
    "state": "states",
}
DATASET_FPS = 20


def dataset_split(task: dict) -> str:
    """Assign scenes by seed's last digit: train 0–7, validation 8–9."""
    seed = task["config"]["data"]["arguments"].get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError(
            f"LIBERO dataset split requires a nonnegative integer seed; got {seed!r}"
        )
    return "train" if seed % 10 < 8 else "validation"


def successful_prefix(task: dict, actions: list[dict]) -> int:
    """Return the prefix length through the first environment success, or zero."""
    result = task.get("result") or {}
    outcome = result.get("data") or {}
    if outcome.get("environment_success") is not True or any(
        key in result.get("missing", {}) for key in ("", "/environment_success")
    ):
        return 0
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


def validate_observation(root: Path, observation: dict[str, Any]) -> None:
    """Check required observation fields; image files are loaded during export."""
    payload = observation.get("payload") or {}
    fields = payload.get("data") or {}
    if payload.get("format") != "libero.observation":
        raise ValueError("Expected a LIBERO observation")
    if observation.get("missing", {}).get("payload") or any(
        key in payload.get("missing", {})
        for key in ("", *(f"/{name}" for name in OBSERVATION_FIELDS.values()))
    ):
        raise ValueError("Missing required LIBERO observation data")
    state = read_array(root, fields.get(OBSERVATION_FIELDS["state"]) or {})
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO state must contain eight finite values")
    for name in ("image", "wrist_image"):
        image = fields.get(OBSERVATION_FIELDS[name]) or {}
        if (
            image.get("$type") not in ("ndarray", "artifact_ref")
            or image.get("shape") != [256, 256, 3]
            or image.get("dtype") != "uint8"
        ):
            raise ValueError(f"Invalid LIBERO {name} array metadata")
        if image["$type"] == "artifact_ref" and (
            image.get("kind") != "ndarray" or not image.get("path")
        ):
            raise ValueError(f"Invalid LIBERO {name} artifact reference")

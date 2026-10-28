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

"""RoboCasa PandaOmron Flywheel conversion for RLDX training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.flywheel.reader import read_array, validate_image

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_FIELDS = {
    f"observation.images.{camera}": camera
    for camera in (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
}
STATE_FIELDS = {
    "end_effector_position_relative": ("robot0_base_to_eef_pos", 3),
    "end_effector_rotation_relative": ("robot0_base_to_eef_quat", 4),
    "gripper_qpos": ("robot0_gripper_qpos", 2),
    "base_position": ("robot0_base_pos", 3),
    "base_rotation": ("robot0_base_quat", 4),
}
ACTION_FIELDS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}
DATASET_FPS = 20
USE_VIDEOS = True


def read_actions(root: Path, payload: dict[str, Any]) -> np.ndarray:
    """Read one executed PandaOmron command in native controller order."""
    payload = payload or {}
    if (
        payload.get("format") != "robocasa.action"
        or payload.get("completeness") != "complete"
        or payload.get("missing")
    ):
        raise ValueError("Missing or unsupported RoboCasa action payload")
    values = read_array(root, payload["data"])
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("RoboCasa actions must contain twelve finite values")
    if not np.isin(values[[6, 11]], (-1, 1)).all():
        raise ValueError("RoboCasa VLA gripper and control mode must be -1 or +1")
    return values


def validate_observation(root: Path, observation: dict[str, Any]) -> np.ndarray:
    """Check required observation fields and return the 16-value RLDX state."""
    payload = observation.get("payload") or {}
    if payload.get("format") != "robocasa.observation":
        raise ValueError("Expected a RoboCasa observation")
    required = (
        "",
        "/images",
        *(f"/{name}" for name, _ in STATE_FIELDS.values()),
        *(f"/images/{camera}" for camera in IMAGE_FIELDS.values()),
    )
    if observation.get("missing", {}).get("payload") or any(
        key in payload.get("missing", {}) for key in required
    ):
        raise ValueError("Missing required RoboCasa observation data")
    fields = payload["data"]
    parts = []
    for name, size in STATE_FIELDS.values():
        values = read_array(root, fields[name])
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"RoboCasa {name} must contain {size} finite values")
        parts.append(values)
    for camera in IMAGE_FIELDS.values():
        validate_image(fields["images"][camera], f"RoboCasa {camera} image")
    return np.concatenate(parts)


def read_frame(
    root: Path, action_record: dict[str, Any], instruction: str
) -> dict[str, Any]:
    """Read one executed command and its before observation in RLDX format."""
    observation = action_record["before_observation"]
    state = validate_observation(root, observation)
    actions = read_actions(root, action_record["action"]).copy()
    # Invert PandaOmronKeyConverter.unmap_action's binary controls.
    actions[[6, 11]] = (actions[[6, 11]] + 1) / 2
    return {
        **{
            name: read_array(root, observation["payload"]["data"]["images"][camera])
            for name, camera in IMAGE_FIELDS.items()
        },
        "observation.state": state,
        "action": actions,
        "prompt": instruction,
    }


def write_metadata(dataset: LeRobotDataset) -> None:
    """Save RLDX field groups and statistics, including quantiles."""
    from lerobot.datasets.utils import serialize_dict, write_json

    modality = {
        "video": {
            camera: {"original_key": name} for name, camera in IMAGE_FIELDS.items()
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    groups = {
        "state": {name: size for name, (_, size) in STATE_FIELDS.items()},
        "action": ACTION_FIELDS,
    }
    for kind, fields in groups.items():
        modality[kind] = {}
        start = 0
        for name, size in fields.items():
            modality[kind][name] = {"start": start, "end": start + size}
            start += size
    statistics = {}
    stored = dataset.hf_dataset.with_format("numpy")
    for name in ("observation.state", "action"):
        statistics[name] = {
            key: dataset.meta.stats[name][key] for key in ("min", "max", "mean", "std")
        }
        q01, q99 = np.quantile(stored[name], (0.01, 0.99), axis=0)
        statistics[name].update(q01=q01, q99=q99)
    write_json(modality, dataset.root / "meta" / "modality.json")
    write_json(serialize_dict(statistics), dataset.root / "meta" / "stats.json")

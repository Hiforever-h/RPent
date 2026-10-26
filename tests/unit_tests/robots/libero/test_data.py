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

import numpy as np
import pytest

from robots.libero.data import (
    dataset_split,
    read_actions,
    successful_prefix,
    validate_observation,
)


def _array(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "$type": "ndarray",
        "dtype": "float32",
        "shape": list(array.shape),
        "value": array.tolist(),
    }


@pytest.mark.parametrize(
    "seed,expected",
    [(seed, "train") for seed in range(8)]
    + [(8, "validation"), (9, "validation")]
    + [
        (10, "train"),
        (25, "train"),
        (38, "validation"),
        (39, "validation"),
        (40, "train"),
        (49, "validation"),
    ],
)
def test_dataset_split_uses_seed_last_digit(seed, expected):
    task = {"config": {"data": {"arguments": {"seed": seed}}}}
    assert dataset_split(task) == expected


@pytest.mark.parametrize("seed", [None, -1, "8", True, 8.0])
def test_missing_or_unsupported_seed_cannot_enter_a_dataset_split(seed):
    task = {"config": {"data": {"arguments": {}}}}
    if seed is not None:
        task["config"]["data"]["arguments"]["seed"] = seed
    with pytest.raises(ValueError, match="requires a nonnegative integer seed"):
        dataset_split(task)


def test_success_prefix_requires_confirmed_task_and_step_results():
    task = {"result": {"data": {"environment_success": True}}}
    actions = [
        {"result": {"format": "libero.step_result", "data": {"terminated": False}}},
        {"result": {"format": "libero.step_result", "data": {"terminated": True}}},
    ]
    assert successful_prefix(task, actions) == 2
    assert successful_prefix(task, actions[:1]) == 0
    actions[0]["result"]["data"]["truncated"] = True
    assert successful_prefix(task, actions) == 0
    actions[0]["result"]["data"]["truncated"] = False
    actions[1]["result"]["missing"] = {"/terminated": {"reason": "capture_failed"}}
    assert successful_prefix(task, actions) == 0
    actions[1]["result"]["missing"] = {"/reward": {"reason": "not_exposed"}}
    assert successful_prefix(task, actions) == 2
    task["result"]["missing"] = {"/environment_success": {"reason": "capture_failed"}}
    assert successful_prefix(task, actions) == 0


@pytest.mark.parametrize(
    "shape,format", [((7,), "libero.action"), ((3, 7), "libero.action_chunk")]
)
def test_native_actions_are_preserved(tmp_path, shape, format):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    restored = read_actions(
        tmp_path, {"format": format, "data": _array(values), "completeness": "complete"}
    )
    np.testing.assert_array_equal(restored, values)
    assert restored.dtype == values.dtype


@pytest.mark.parametrize("values", [[0] * 6, [np.inf] * 7, [np.nan] * 7, [[0] * 7]])
def test_invalid_native_action_is_rejected(tmp_path, values):
    with pytest.raises(ValueError, match="seven finite values"):
        read_actions(
            tmp_path,
            {
                "format": "libero.action",
                "data": _array(values),
                "completeness": "complete",
            },
        )


@pytest.fixture
def observation():
    return {
        "payload": {
            "format": "libero.observation",
            "data": {
                "states": _array([0] * 8),
                **{
                    name: {
                        "$type": "artifact_ref",
                        "kind": "ndarray",
                        "path": f"artifacts/{name}.npy",
                        "dtype": "uint8",
                        "shape": [256, 256, 3],
                    }
                    for name in ("main_images", "wrist_images")
                },
            },
        },
    }


def test_missing_unused_observation_fields_do_not_drop_training_data(
    tmp_path, observation
):
    observation["payload"]["completeness"] = "partial"
    observation["payload"]["missing"] = {
        "/extra_view_images": {"reason": "not_exposed"}
    }
    validate_observation(tmp_path, observation)


@pytest.mark.parametrize("field", ["states", "main_images", "wrist_images"])
def test_missing_required_observation_field_is_rejected(tmp_path, observation, field):
    observation["payload"]["data"][field] = None
    with pytest.raises(ValueError):
        validate_observation(tmp_path, observation)


@pytest.mark.parametrize("state", [[0] * 7, [np.inf] * 8, [np.nan] * 8])
def test_invalid_proprioception_is_rejected(tmp_path, observation, state):
    observation["payload"]["data"]["states"] = _array(state)
    with pytest.raises(ValueError, match="eight finite values"):
        validate_observation(tmp_path, observation)


@pytest.mark.parametrize(
    "field,value",
    [("shape", [128, 128, 3]), ("dtype", "float32"), ("kind", "bytes"), ("path", "")],
)
def test_invalid_image_metadata_is_rejected(tmp_path, observation, field, value):
    observation["payload"]["data"]["main_images"][field] = value
    with pytest.raises(ValueError, match="Invalid LIBERO image"):
        validate_observation(tmp_path, observation)

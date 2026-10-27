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

from copy import deepcopy

import numpy as np
import pytest

from rpent.data.prepare import iter_samples, select_vla_segments
from rpent.data.reader import read_array


def _ranges(result):
    return [
        (segment["start_index"], segment["stop_index"])
        for segment in result["segments"]
    ]


def test_selects_executed_predictions_across_chunks_without_loading_images(trajectory):
    original = deepcopy(trajectory)
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    result = select_vla_segments(trajectory)

    assert _ranges(result) == [(1, 7)]
    segment = result["segments"][0]
    assert segment["tool_call_id"] == trajectory["predictions"][0]["tool_call_id"]
    assert segment["instruction"] == "pick up the bowl"
    np.testing.assert_array_equal(
        segment["actions"],
        np.stack(
            [
                read_array(trajectory["root"], action["action"]["data"])
                for action in trajectory["actions"][1:7]
            ]
        ),
    )
    assert result["rejected"] == []
    assert trajectory == original


@pytest.mark.parametrize("success", [False, None, "true", 1])
def test_planner_completion_does_not_establish_environment_success(trajectory, success):
    trajectory["task"]["result"]["data"]["environment_success"] = success
    result = select_vla_segments(trajectory)
    assert result["segments"] == []
    assert result["rejected"][0]["reason"] == "no_confirmed_environment_success"


def test_includes_success_action_but_excludes_later_chunk_padding(trajectory):
    trajectory["actions"][4]["result"]["data"]["terminated"] = True
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 5)]
    assert [
        row["action_id"]
        for row in result["rejected"]
        if row["reason"] == "after_environment_success"
    ] == [action["action_id"] for action in trajectory["actions"][5:7]]


@pytest.mark.parametrize("source", ["direct", "tool_computed"])
def test_non_vla_action_splits_the_segment_even_when_prediction_claims_it(
    trajectory, source
):
    trajectory["actions"][2]["source"] = source
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


def test_vla_actions_without_a_prediction_are_excluded(trajectory):
    trajectory["predictions"].pop(0)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert result["rejected"] == []


def test_unexecuted_prediction_does_not_supply_training_actions(trajectory):
    prediction = deepcopy(trajectory["predictions"][0])
    prediction.update(prediction_id="V-unexecuted", action_ids=[])
    trajectory["predictions"].append(prediction)
    assert _ranges(select_vla_segments(trajectory)) == [(1, 7)]


@pytest.mark.parametrize("field", ["instruction", "tool_call_id"])
def test_instruction_or_tool_call_change_starts_a_new_segment(trajectory, field):
    trajectory["predictions"][1][field] = "different"
    if field == "tool_call_id":
        for action in trajectory["actions"][4:7]:
            action[field] = "different"
    assert _ranges(select_vla_segments(trajectory)) == [(1, 4), (4, 7)]


def test_missing_middle_action_breaks_continuity_without_discarding_other_actions(
    trajectory,
):
    missing_id = trajectory["actions"].pop(2)["action_id"]
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (2, 6)]
    assert any(
        row.get("action_id") == missing_id and row["reason"] == "missing_action_record"
        for row in result["rejected"]
    )


@pytest.mark.parametrize("change", ["id", "payload"])
def test_disconnected_observations_are_not_joined(trajectory, change):
    action = trajectory["actions"][2]
    if change == "id":
        action["before_observation_id"] = "O-disconnected"
        action["before_observation"]["observation_id"] = "O-disconnected"
    else:
        action["before_observation"]["payload"]["data"]["states"]["value"][0] += 1
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (2, 7)]


@pytest.mark.parametrize("field", ["before_observation", "after_observation", "action"])
def test_missing_required_action_data_excludes_the_step_and_breaks_continuity(
    trajectory, field
):
    trajectory["actions"][2][field] = None
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


def test_action_record_cannot_contain_an_entire_chunk(trajectory):
    payload = trajectory["actions"][2]["action"]
    payload["format"] = "libero.action_chunk"
    payload["data"]["shape"] = [1, 7]
    payload["data"]["value"] = [payload["data"]["value"]]
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (3, 7)]
    assert any(
        "Expected one executed action" in row["reason"] for row in result["rejected"]
    )


@pytest.mark.parametrize("index,expected", [(1, [(4, 7)]), (2, [(1, 2), (3, 7)])])
def test_incomplete_before_observation_excludes_the_prediction_or_step(
    trajectory, index, expected
):
    observation = trajectory["actions"][index]["before_observation"]
    observation["payload"]["data"].pop("main_images")
    trajectory["actions"][index - 1]["after_observation"] = deepcopy(observation)
    if index == 1:
        trajectory["predictions"][0]["input_observation"] = deepcopy(observation)
    assert _ranges(select_vla_segments(trajectory)) == expected


@pytest.mark.parametrize("trajectory", [35], indirect=True)
@pytest.mark.parametrize("field", ["main_images", "states"])
def test_final_after_observation_data_is_not_required_for_samples(trajectory, field):
    observation = trajectory["actions"][35]["after_observation"]
    observation["payload"]["data"].pop(field)
    trajectory["actions"][36]["before_observation"] = deepcopy(observation)
    result = select_vla_segments(trajectory)

    assert _ranges(result) == [(1, 36)]
    assert result["rejected"] == []
    assert len(list(iter_samples(trajectory, result["segments"]))) == 26


def test_legacy_executed_action_must_equal_its_prediction(legacy_trajectory):
    trajectory = legacy_trajectory
    trajectory["actions"][2]["action"]["data"]["value"][0] += 1
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 2), (3, 7)]
    assert any(
        "differs from its prediction" in row["reason"] for row in result["rejected"]
    )


def test_legacy_prediction_action_count_must_match_references(legacy_trajectory):
    prediction = legacy_trajectory["predictions"][0]
    array = prediction["actions"]["data"]
    array["value"].append(array["value"][-1])
    array["shape"][0] += 1
    result = select_vla_segments(legacy_trajectory)
    assert _ranges(result) == [(4, 7)]
    assert any("different lengths" in row["reason"] for row in result["rejected"])


def test_legacy_predictions_still_supply_complete_training_samples(legacy_trajectory):
    result = select_vla_segments(legacy_trajectory)
    assert legacy_trajectory["schema_version"] == 3
    assert _ranges(result) == [(1, 7)]
    assert result["rejected"] == []
    expected = np.concatenate(
        [
            read_array(legacy_trajectory["root"], row["actions"]["data"])
            for row in legacy_trajectory["predictions"]
        ]
    )
    samples = list(iter_samples(legacy_trajectory, result["segments"], horizon=2))
    assert len(samples) == 5
    for offset, sample in enumerate(samples):
        assert sample["actions"].dtype == np.float32
        np.testing.assert_array_equal(sample["actions"], expected[offset : offset + 2])


def test_action_from_another_tool_call_is_excluded(trajectory):
    trajectory["actions"][2]["tool_call_id"] = "T-other"
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


def test_missing_observation_mark_is_respected_even_if_reference_exists(trajectory):
    trajectory["actions"][2]["missing"]["before_observation_id"] = {
        "reason": "capture_failed"
    }
    assert _ranges(select_vla_segments(trajectory)) == [(1, 2), (3, 7)]


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_instruction",
        "wrong_input",
        "reverse_actions",
        "missing_input",
    ],
)
def test_invalid_prediction_does_not_supply_training_actions(trajectory, corruption):
    prediction = trajectory["predictions"][0]
    if corruption == "empty_instruction":
        prediction["instruction"] = " "
    elif corruption == "wrong_input":
        prediction["input_observation_id"] = "O-wrong"
    elif corruption == "reverse_actions":
        prediction["action_ids"][1:] = reversed(prediction["action_ids"][1:])
    else:
        prediction["missing"]["input_observation_id"] = {"reason": "capture_failed"}
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert any(
        row.get("prediction_id") == prediction["prediction_id"]
        for row in result["rejected"]
    )


def test_two_predictions_cannot_claim_the_same_executed_action(trajectory):
    duplicate = deepcopy(trajectory["predictions"][0])
    duplicate["prediction_id"] = "V-duplicate"
    trajectory["predictions"].append(duplicate)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(4, 7)]
    assert {
        (row["prediction_id"], row["action_id"])
        for row in result["rejected"]
        if row["reason"] == "multiple_predictions_claim_action"
    } == {
        (prediction["prediction_id"], action_id)
        for prediction in (trajectory["predictions"][0], duplicate)
        for action_id in prediction["action_ids"]
    }


def test_predictions_follow_execution_order_when_records_arrive_out_of_order(
    trajectory,
):
    trajectory["predictions"].reverse()
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 7)]
    np.testing.assert_array_equal(
        result["segments"][0]["actions"],
        np.arange(7, 49, dtype=np.float32).reshape(6, 7) / 100,
    )


def test_script_between_predictions_breaks_the_segment_without_being_a_candidate(
    trajectory,
):
    script = deepcopy(trajectory["actions"][0])
    observation = trajectory["actions"][4]["before_observation"]
    script.update(
        action_id="A-script",
        before_observation_id=observation["observation_id"],
        after_observation_id=observation["observation_id"],
        before_observation=observation,
        after_observation=observation,
    )
    trajectory["actions"].insert(4, script)
    result = select_vla_segments(trajectory)
    assert _ranges(result) == [(1, 4), (5, 8)]
    assert result["rejected"] == []
    assert list(iter_samples(trajectory, result["segments"], horizon=4)) == []


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_segment_and_sample_action_values_and_dtype_come_from_executed_actions(
    trajectory,
):
    for action in trajectory["actions"]:
        action["action"]["data"]["dtype"] = "float64"
    result = select_vla_segments(trajectory)
    expected = np.stack(
        [
            read_array(trajectory["root"], action["action"]["data"])
            for action in trajectory["actions"][1:36]
        ]
    )
    segment = result["segments"][0]
    assert segment["actions"].dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(segment["actions"], expected)
    samples = list(iter_samples(trajectory, result["segments"]))
    assert len(samples) == 26
    for offset, sample in enumerate(samples):
        assert sample["actions"].dtype == np.float64
        np.testing.assert_array_equal(sample["actions"], expected[offset : offset + 10])


@pytest.mark.parametrize(
    "records,field", [("actions", "action_id"), ("predictions", "prediction_id")]
)
def test_duplicate_record_ids_fail_with_a_useful_error(trajectory, records, field):
    trajectory[records][1][field] = trajectory[records][0][field]
    with pytest.raises(ValueError, match=f"[Dd]uplicate {field}"):
        select_vla_segments(trajectory)


def test_robot_without_data_support_has_a_clear_error(trajectory):
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = "not_installed"
    with pytest.raises(ValueError, match="not supported for not_installed"):
        select_vla_segments(trajectory)


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_samples_use_each_start_observation_local_instruction_and_next_ten_actions(
    trajectory,
):
    original = deepcopy(trajectory)
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"])
    )

    assert len(samples) == 26
    for start, sample in enumerate(samples, 1):
        before = trajectory["actions"][start]["before_observation"]
        assert sample["root"] == trajectory["root"]
        assert sample["task_id"] == trajectory["task"]["task_id"]
        assert sample["tool_call_id"] == trajectory["predictions"][0]["tool_call_id"]
        assert (sample["start_index"], sample["stop_index"]) == (start, start + 10)
        assert sample["observation_id"] == before["observation_id"]
        assert sample["instruction"] == "pick up the bowl"
        assert sample["images"] == {
            "image": before["payload"]["data"]["main_images"],
            "wrist_image": before["payload"]["data"]["wrist_images"],
        }
        np.testing.assert_array_equal(
            sample["state"], np.full(8, start, dtype=np.float32)
        )
        expected = (
            np.arange(start * 7, (start + 10) * 7, dtype=np.float32).reshape(10, 7)
            / 100
        )
        np.testing.assert_array_equal(sample["actions"], expected)
        assert sample["state"].dtype == sample["actions"].dtype == np.float32
    np.testing.assert_array_equal(
        read_array(samples[0]["root"], samples[0]["images"]["image"]),
        np.zeros((256, 256, 3), dtype=np.uint8),
    )
    samples[0]["actions"][:] = -1
    assert samples[1]["actions"][0, 0] != -1
    assert trajectory == original


@pytest.mark.parametrize(
    "trajectory,expected_count",
    [(0, 0), (9, 0), (10, 1), (11, 2)],
    indirect=["trajectory"],
)
def test_short_tails_do_not_produce_padded_samples(trajectory, expected_count):
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"])
    )
    assert len(samples) == expected_count
    assert all(sample["actions"].shape == (10, 7) for sample in samples)


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_samples_keep_image_references_until_images_are_loaded(trajectory):
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"])
    )
    assert len(samples) == 26
    with pytest.raises(FileNotFoundError):
        read_array(samples[0]["root"], samples[0]["images"]["image"])


@pytest.mark.parametrize("trajectory", [20], indirect=True)
@pytest.mark.parametrize(
    "boundary", ["script", "missing_action", "instruction", "tool_call_id"]
)
def test_sample_windows_do_not_cross_segment_boundaries(trajectory, boundary):
    if boundary == "script":
        trajectory["actions"][10]["source"] = "tool_computed"
        expected_starts = [11]
    elif boundary == "missing_action":
        trajectory["actions"].pop(8)
        expected_starts = [8, 9, 10]
    else:
        for prediction in trajectory["predictions"][3:]:
            prediction[boundary] = "different"
        if boundary == "tool_call_id":
            for action in trajectory["actions"][10:21]:
                action[boundary] = "different"
        expected_starts = [10, 11]
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"])
    )
    assert [sample["start_index"] for sample in samples] == expected_starts


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_samples_end_at_environment_success_and_exclude_failed_tasks(trajectory):
    trajectory["actions"][19]["result"]["data"]["terminated"] = True
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"])
    )
    assert len(samples) == 10
    assert samples[-1]["stop_index"] == 20
    trajectory["task"]["result"]["data"]["environment_success"] = False
    assert (
        list(iter_samples(trajectory, select_vla_segments(trajectory)["segments"]))
        == []
    )


def test_sample_horizon_can_be_configured(trajectory):
    samples = list(
        iter_samples(trajectory, select_vla_segments(trajectory)["segments"], horizon=4)
    )
    assert [sample["start_index"] for sample in samples] == [1, 2, 3]
    assert all(sample["actions"].shape == (4, 7) for sample in samples)


@pytest.mark.parametrize("trajectory", [35], indirect=True)
@pytest.mark.parametrize(
    "seed,expected",
    [(7, "train"), (8, "validation"), (37, "train"), (48, "validation")],
)
def test_repeated_scene_attempts_keep_all_overlapping_samples_in_one_split(
    trajectory, seed, expected
):
    trajectory["task"]["config"]["data"]["arguments"]["seed"] = seed
    for attempt in (1, 0):
        recording = deepcopy(trajectory)
        recording["task"]["task_id"] = f"attempt-{attempt}"
        original = deepcopy(recording)
        samples = list(
            iter_samples(recording, select_vla_segments(recording)["segments"])
        )
        assert len(samples) == 26
        assert {sample["split"] for sample in samples} == {expected}
        assert recording == original


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_sample_horizon_must_be_a_positive_integer(trajectory, horizon):
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_samples(trajectory, [], horizon=horizon))

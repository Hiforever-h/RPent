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

import json
import shutil

import numpy as np
import pytest

from rpent.data.prepare import export_dataset, load_dataset

pytest.importorskip("lerobot")


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_export_stores_complete_samples_in_independent_splits(trajectory, tmp_path):
    jobs = tmp_path / "jobs"
    for name, seed in (("train/attempt_01", 7), ("train-second", 6), ("validation", 8)):
        path = shutil.copytree(trajectory["root"], jobs / name)
        document = json.loads((path / "task.json").read_text())
        document["task"]["config"]["data"]["arguments"]["seed"] = seed
        (path / "task.json").write_text(json.dumps(document))
    destination = tmp_path / "export"
    report = export_dataset(jobs, destination)
    assert report["episode_counts"] == {"train": 2, "validation": 1}
    assert report["sample_counts"] == {"train": 52, "validation": 26}
    assert report["trajectory_count"] == report["exported_trajectory_count"] == 3
    assert report["fps"] == 20
    assert not list(destination.rglob("*.png"))
    assert len(list(destination.rglob("*.parquet"))) == 3
    assert not list(destination.glob("*_samples.jsonl"))
    assert [row["excluded_action_count"] for row in report["trajectories"]] == [2, 2, 2]
    assert all(
        row["excluded_action_counts"]
        == {"non_vla": 2, "unselected_vla": 0, "short_segment": 0}
        for row in report["trajectories"]
    )

    # Embedded images and relative dataset paths must survive moving the export.
    moved = destination.rename(tmp_path / "moved")
    shutil.rmtree(jobs)
    for split, count in (("train", 52), ("validation", 26)):
        dataset = load_dataset(moved, split)
        assert len(dataset) == count
        assert dataset.meta.total_frames == count
        assert dataset.features["actions"]["shape"] == (10, 7)
        for offset in range(count):
            sample = dataset[offset]
            start = offset % 26 + 1
            expected = (
                np.arange(start * 7, (start + 10) * 7, dtype=np.float32).reshape(10, 7)
                / 100
            )
            np.testing.assert_array_equal(sample["actions"], expected)
            np.testing.assert_array_equal(
                sample["state"], np.full(8, start, dtype=np.float32)
            )
            assert "actions_is_pad" not in sample
            assert sample["image"].shape == sample["wrist_image"].shape == (3, 256, 256)
            assert sample["prompt"] == sample["task"] == "pick up the bowl"
            assert sample["episode_index"].item() == offset // 26


@pytest.mark.parametrize("trajectory", [35], indirect=True)
def test_export_omits_short_segments_and_never_bridges_a_script(trajectory, tmp_path):
    path = trajectory["root"] / "actions.jsonl"
    actions = [json.loads(line) for line in path.read_text().splitlines()]
    actions[8]["source"] = "tool_computed"
    path.write_text("".join(json.dumps(action) + "\n" for action in actions))

    output = tmp_path / "split-export"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 1, "validation": 0}
    assert report["sample_counts"] == {"train": 18, "validation": 0}
    source = report["trajectories"][0]
    assert source["selected_action_count"] == 34
    assert source["excluded_action_count"] == 10
    assert source["excluded_action_counts"] == {
        "non_vla": 3,
        "unselected_vla": 0,
        "short_segment": 7,
    }
    assert any(row["reason"] == "shorter_than_horizon" for row in source["rejected"])
    assert source["episodes"][0]["start_index"] == 9
    assert source["episodes"][0]["stop_index"] == 36
    assert [
        sample["state"][0].item() for sample in load_dataset(output, "train")
    ] == list(range(9, 27))
    assert load_dataset(output, "train")[0]["state"][0] == 9
    assert len(load_dataset(output, "validation")) == 0


@pytest.mark.parametrize("trajectory", [10], indirect=True)
def test_missing_image_aborts_export_without_publishing_partial_data(
    trajectory, tmp_path
):
    for path in trajectory["root"].rglob("*.npy"):
        path.unlink()
    output = tmp_path / "missing-image"
    with pytest.raises(FileNotFoundError):
        export_dataset(trajectory["root"], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".missing-image-*"))


def test_export_does_not_overwrite_an_existing_directory(trajectory, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        export_dataset(trajectory["root"], output)
    assert marker.read_text() == "keep"


def test_failed_trajectories_still_produce_an_exclusion_report(trajectory, tmp_path):
    path = trajectory["root"] / "task.json"
    document = json.loads(path.read_text())
    document["task"]["result"]["data"]["environment_success"] = False
    path.write_text(json.dumps(document))
    output = tmp_path / "failed-only"
    report = export_dataset(trajectory["root"], output)
    assert report["episode_counts"] == {"train": 0, "validation": 0}
    assert report["dataset_paths"] == {"train": None, "validation": None}
    assert report["trajectories"][0]["excluded_action_count"] == 8
    assert (
        report["trajectories"][0]["rejected"][0]["reason"]
        == "no_confirmed_environment_success"
    )
    assert json.loads((output / "report.json").read_text())["sample_counts"] == {
        "train": 0,
        "validation": 0,
    }
    assert len(load_dataset(output, "train")) == 0

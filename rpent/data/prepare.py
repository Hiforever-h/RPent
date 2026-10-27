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

"""Select executed VLA actions, build samples and save training datasets."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rpent.data.reader import iter_trajectories, read_array

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def iter_samples(
    trajectory: dict[str, Any], segments: list[dict], horizon: int = 10
) -> Iterator[dict[str, Any]]:
    """Yield overlapping windows from segments returned by select_vla_segments.

    The robot assigns one train/validation split to the entire trajectory.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not segments:
        return
    root = trajectory["root"]
    task = trajectory["task"]
    data = import_module(
        f"robots.{task['config']['data']['arguments']['robot_name']}.data"
    )
    split = data.dataset_split(task)
    for segment in segments:
        start, stop = segment["start_index"], segment["stop_index"]
        if stop - start < horizon:
            continue
        records = trajectory["actions"][start:stop]
        actions = segment["actions"]
        for offset in range(len(records) - horizon + 1):
            observation = records[offset]["before_observation"]
            fields = observation["payload"]["data"]
            yield {
                "root": root,
                "task_id": task["task_id"],
                "split": split,
                "tool_call_id": segment["tool_call_id"],
                "start_index": start + offset,
                "stop_index": start + offset + horizon,
                "observation_id": observation["observation_id"],
                "images": {
                    name: fields[field]
                    for name, field in data.OBSERVATION_FIELDS.items()
                    if name != "state"
                },
                "state": read_array(root, fields[data.OBSERVATION_FIELDS["state"]]),
                "instruction": segment["instruction"],
                "actions": actions[offset : offset + horizon].copy(),
            }


def select_vla_segments(trajectory: dict[str, Any]) -> dict[str, list[dict]]:
    """Return continuous ranges of executed VLA actions and exclusion reasons."""
    task = trajectory["task"]
    robot = task["config"]["data"]["arguments"].get("robot_name")
    if not isinstance(robot, str) or not robot.isidentifier():
        raise ValueError("Task config must identify a robot_name")
    module_name = f"robots.{robot}.data"
    try:
        data = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in (f"robots.{robot}", module_name):
            raise ValueError(
                f"Trajectory data selection is not supported for {robot}"
            ) from exc
        raise

    actions = trajectory["actions"]
    stop = data.successful_prefix(task, actions)
    segments = []
    rejected = []
    if not stop:
        return {
            "segments": [],
            "rejected": [
                {
                    "task_id": task["task_id"],
                    "reason": "no_confirmed_environment_success",
                }
            ],
        }

    positions = {}
    for index, action in enumerate(actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"Action at index {index} has no action_id")
        if action_id in positions:
            raise ValueError(f"Duplicate action_id: {action_id}")
        positions[action_id] = index

    predictions = []
    references = Counter()
    seen_predictions = set()
    for prediction in trajectory["predictions"]:
        prediction_id = prediction.get("prediction_id")
        if (
            not isinstance(prediction_id, str)
            or not prediction_id
            or prediction_id in seen_predictions
        ):
            raise ValueError(f"Missing or duplicate prediction_id: {prediction_id}")
        seen_predictions.add(prediction_id)
        try:
            action_ids = prediction["action_ids"]
            if not isinstance(action_ids, list) or any(
                not isinstance(item, str) for item in action_ids
            ):
                raise ValueError("Prediction action_ids must be a list of IDs")
            if not action_ids or not any(
                positions.get(item, stop) < stop for item in action_ids
            ):
                continue
            if len(set(action_ids)) != len(action_ids):
                raise ValueError("Prediction repeats an action_id")
            instruction = prediction.get("instruction")
            if (
                not isinstance(instruction, str)
                or not instruction.strip()
                or not prediction.get("tool_call_id")
            ):
                raise ValueError("Missing local instruction or tool call")
            if any(
                key in prediction.get("missing", {})
                for key in (
                    "actions",
                    "action_ids",
                    "instruction",
                    "input_observation_id",
                )
            ):
                raise ValueError("Missing prediction data")
            values = None
            if trajectory["schema_version"] == 3:
                values = data.read_actions(trajectory["root"], prediction["actions"])
                if values.ndim != 2 or len(values) != len(action_ids):
                    raise ValueError(
                        "Prediction actions and action_ids have different lengths"
                    )
            observation = prediction.get("input_observation")
            observation_id = prediction.get("input_observation_id")
            first = positions.get(action_ids[0])
            if (
                not observation_id
                or not isinstance(observation, dict)
                or observation.get("observation_id") != observation_id
                or first is None
                or trajectory["actions"][first].get("before_observation_id")
                != observation_id
                or trajectory["actions"][first].get("before_observation") != observation
            ):
                raise ValueError(
                    "Prediction input does not match its first executed action"
                )
            data.validate_observation(trajectory["root"], observation)
            known_positions = [
                positions[item] for item in action_ids if item in positions
            ]
            if known_positions != sorted(known_positions):
                raise ValueError("Prediction action order differs from execution order")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            rejected.append(
                {"prediction_id": prediction_id, "reason": f"invalid_prediction: {exc}"}
            )
            continue
        predictions.append((prediction, values))
        references.update(action_ids)

    segment = previous = None
    for prediction, values in sorted(
        predictions, key=lambda item: positions[item[0]["action_ids"][0]]
    ):
        for offset, action_id in enumerate(prediction["action_ids"]):
            index = positions.get(action_id)
            reason = None
            if index is None:
                reason = "missing_action_record"
            elif index >= stop:
                reason = "after_environment_success"
            elif references[action_id] > 1:
                reason = "multiple_predictions_claim_action"
            else:
                action = actions[index]
                try:
                    if action.get("source") != "vla":
                        raise ValueError("Prediction references a non-VLA action")
                    if action.get("tool_call_id") != prediction["tool_call_id"]:
                        raise ValueError(
                            "Action and prediction belong to different tool calls"
                        )
                    if any(
                        key in action.get("missing", {})
                        for key in (
                            "action",
                            "before_observation_id",
                            "after_observation_id",
                        )
                    ):
                        raise ValueError("Missing action or observation reference")
                    for side in ("before", "after"):
                        observation = action.get(f"{side}_observation")
                        observation_id = action.get(f"{side}_observation_id")
                        if (
                            not observation_id
                            or not isinstance(observation, dict)
                            or observation.get("observation_id") != observation_id
                        ):
                            raise ValueError(
                                f"Missing or inconsistent {side} observation"
                            )
                        # The first before-observation was checked with the prediction.
                        if (
                            side == "before"
                            and action_id != prediction["action_ids"][0]
                        ):
                            data.validate_observation(trajectory["root"], observation)
                    actual = data.read_actions(trajectory["root"], action["action"])
                    if actual.ndim != 1:
                        raise ValueError("Expected one executed action")
                    if values is not None and not np.array_equal(
                        actual, values[offset]
                    ):
                        raise ValueError("Executed action differs from its prediction")
                    value = actual if values is None else values[offset]
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    reason = f"invalid_action: {exc}"

            if reason is not None:
                rejected.append(
                    {
                        "prediction_id": prediction["prediction_id"],
                        "action_id": action_id,
                        "reason": reason,
                    }
                )
                segment = previous = None
                continue

            consecutive = (
                segment is not None
                and segment["stop_index"] == index
                and segment["tool_call_id"] == prediction["tool_call_id"]
                and segment["instruction"] == prediction["instruction"]
                and previous["after_observation"] == action["before_observation"]
            )
            if not consecutive:
                segment = {
                    "start_index": index,
                    "stop_index": index,
                    "tool_call_id": prediction["tool_call_id"],
                    "instruction": prediction["instruction"],
                    "actions": [],
                }
                segments.append(segment)
            segment["actions"].append(value)
            segment["stop_index"] = index + 1
            previous = action

    for segment in segments:
        segment["actions"] = np.stack(segment["actions"])
    return {"segments": segments, "rejected": rejected}


def export_dataset(
    input_root: Path | str,
    output_dir: Path | str,
    *,
    pattern: str = "*",
    horizon: int = 10,
    repo_id: str = "rpent/vla-sft",
) -> dict[str, Any]:
    """Export complete samples into separate train and validation datasets."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report = {
        "schema_version": 2,
        "repo_id": repo_id,
        "dataset_paths": {"train": None, "validation": None},
        "horizon": horizon,
        "robot": None,
        "fps": None,
        "episode_counts": {"train": 0, "validation": 0},
        "sample_counts": {"train": 0, "validation": 0},
        "trajectories": [],
    }
    datasets = {"train": None, "validation": None}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as tmp:
        root = Path(tmp)
        for trajectory in iter_trajectories(input_root, pattern):
            selected = select_vla_segments(trajectory)
            source = {
                "source": str(trajectory["root"]),
                "task_id": trajectory["task"]["task_id"],
                "action_count": len(trajectory["actions"]),
                "selected_action_count": sum(
                    len(segment["actions"]) for segment in selected["segments"]
                ),
                "exported_action_count": 0,
                "sample_counts": {"train": 0, "validation": 0},
                "episodes": [],
                "rejected": selected["rejected"],
            }
            report["trajectories"].append(source)
            for segment in selected["segments"]:
                if len(segment["actions"]) < horizon:
                    source["rejected"].append(
                        {
                            "start_index": segment["start_index"],
                            "stop_index": segment["stop_index"],
                            "reason": "shorter_than_horizon",
                        }
                    )
                    continue
                task = trajectory["task"]
                robot = task["config"]["data"]["arguments"]["robot_name"]
                if report["robot"] is not None and robot != report["robot"]:
                    raise ValueError("Export each robot into a separate dataset")
                for sample in iter_samples(trajectory, [segment], horizon=horizon):
                    split = sample["split"]
                    dataset = datasets[split]
                    frame = {
                        name: read_array(trajectory["root"], value)
                        for name, value in sample["images"].items()
                    }
                    frame.update(state=sample["state"], actions=sample["actions"])
                    if dataset is None:
                        data = import_module(f"robots.{robot}.data")
                        features = {
                            name: {
                                "dtype": "image"
                                if name in sample["images"]
                                else str(value.dtype),
                                "shape": value.shape,
                                "names": ["height", "width", "channel"]
                                if name in sample["images"]
                                else None,
                            }
                            for name, value in frame.items()
                        }
                        features["prompt"] = {
                            "dtype": "string",
                            "shape": (1,),
                            "names": None,
                        }
                        dataset = LeRobotDataset.create(
                            repo_id=repo_id,
                            root=root / split,
                            robot_type=robot,
                            fps=data.DATASET_FPS,
                            features=features,
                            use_videos=False,
                            video_backend="pyav",
                        )
                        datasets[split] = dataset
                        report["dataset_paths"][split] = split
                        report.update(robot=robot, fps=data.DATASET_FPS)
                    frame["prompt"] = sample["instruction"]
                    dataset.add_frame(frame, task=sample["instruction"])
                    report["sample_counts"][split] += 1
                    source["sample_counts"][split] += 1
                dataset.save_episode()
                source["episodes"].append(
                    {
                        "split": split,
                        "episode_index": report["episode_counts"][split],
                        "tool_call_id": segment["tool_call_id"],
                        "start_index": segment["start_index"],
                        "stop_index": segment["stop_index"],
                    }
                )
                report["episode_counts"][split] += 1
                source["exported_action_count"] += len(segment["actions"])
            source["excluded_action_count"] = (
                source["action_count"] - source["exported_action_count"]
            )
            non_vla_count = sum(
                action.get("source") != "vla" for action in trajectory["actions"]
            )
            source["excluded_action_counts"] = {
                "non_vla": non_vla_count,
                "unselected_vla": source["action_count"]
                - non_vla_count
                - source["selected_action_count"],
                "short_segment": source["selected_action_count"]
                - source["exported_action_count"],
            }
        report["trajectory_count"] = len(report["trajectories"])
        report["exported_trajectory_count"] = sum(
            source["exported_action_count"] > 0 for source in report["trajectories"]
        )
        # Image bytes are embedded in Parquet; remove LeRobot's staging PNGs.
        for split, dataset in datasets.items():
            if dataset is not None:
                shutil.rmtree(root / split / "images")
        (root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        root.rename(destination)
    return {"output_dir": str(destination), **report}


def load_dataset(
    output_dir: Path | str, split: Literal["train", "validation"]
) -> LeRobotDataset | list[dict[str, Any]]:
    """Read one split with complete action windows, without temporal lookups."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    root = Path(output_dir).expanduser().resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["sample_counts"][split] == 0:
        return []
    return LeRobotDataset(
        report["repo_id"],
        root=root / report["dataset_paths"][split],
        video_backend="pyav",
    )

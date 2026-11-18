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
"""Offline unit tests for .agents/skills/review-pr/lint_review.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def lint_review():
    script = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "review-pr"
        / "lint_review.py"
    )
    spec = importlib.util.spec_from_file_location("lint_review", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lint_review"] = module
    spec.loader.exec_module(module)
    return module


def _valid_body(lint_review, recommendation: str = "- Ready for senior review") -> str:
    sections = []
    for i, q in enumerate(lint_review.QUESTIONS, 1):
        content = recommendation if i == len(lint_review.QUESTIONS) else "- Checked."
        sections.append(f"### {i}. {q}\n\n{content}")
    return "\n\n".join(sections)


def test_markdown_unfenced_and_wrapped_handle_nested_fences(lint_review):
    text = (
        "First line of a paragraph\n"
        "second line wrapped by mistake.\n"
        "\n"
        "````markdown\n"
        "Inside outer fence\n"
        "```python\n"
        "x = 1\n"
        "```\n"
        "Still inside outer fence\n"
        "````\n"
        "\n"
        "- A list item\n"
        "- Another list item\n"
    )
    unfenced_lines = [line for _, line in lint_review.Markdown.unfenced(text)]
    assert "Inside outer fence" not in unfenced_lines
    assert lint_review.Markdown.wrapped(text) == [2]


def test_parse_patch_maps_left_and_right_hunks(lint_review):
    patch = (
        "@@ -10,4 +10,5 @@\n"
        " context\n"
        "-old_line\n"
        "+new_line_1\n"
        "+new_line_2\n"
        " trailing\n"
        "\\ No newline at end of file\n"
        "@@ -30,2 +31,2 @@\n"
        "-old_second\n"
        "+new_second\n"
    )
    left: dict[int, int] = {}
    right: dict[int, int] = {}
    lint_review.PullRequest.parse_patch(patch, left, right)
    assert left == {10: 0, 11: 0, 12: 0, 30: 1}
    assert right == {10: 0, 11: 0, 12: 0, 13: 0, 31: 1}


def test_linter_accepts_valid_payload_and_checks_recommendation_verdict(lint_review):
    # Mentioning lowercase "blocked" inside prose must not trigger a multi-verdict warning.
    body = _valid_body(
        lint_review,
        recommendation="- Changes requested — fix the blocked shutdown path.",
    )
    payload = {
        "commit_id": "a" * 40,
        "event": "COMMENT",
        "body": body,
        "comments": [
            {
                "path": "rpent/x.py",
                "line": 11,
                "body": "The call leaks a daemon.\n\nSuggested fix: close the daemon.\n\n```suggestion\nclose()\n```",
            }
        ],
    }
    findings = lint_review.ReviewLinter(payload, None).run()
    assert findings == []


def test_linter_rejects_lowercase_blocked_prose_without_verdict(lint_review):
    body = _valid_body(lint_review, recommendation="- Merge is not blocked by CI.")
    payload = {"body": body, "comments": []}
    findings = lint_review.ReviewLinter(payload, None).run()
    assert any(
        f.level == "ERROR" and "must name one recommendation" in f.message
        for f in findings
    )


def test_linter_rejects_null_or_malformed_top_level_fields(lint_review):
    payload = {
        "commit_id": None,
        "event": "COMMENT",
        "body": None,
        "comments": None,
    }
    findings = lint_review.ReviewLinter(payload, None).run()
    messages = [f.message for f in findings if f.level == "ERROR"]
    assert any("commit_id must be a 40-character hex SHA" in m for m in messages)
    assert any("body must be a string" in m for m in messages)
    assert any("comments must be a list of comment objects" in m for m in messages)


def test_linter_requires_matching_start_side_on_multiline_left_comment(lint_review):
    body = _valid_body(lint_review)
    comment_missing_start_side = {
        "path": "rpent/x.py",
        "start_line": 10,
        "line": 11,
        "side": "LEFT",
        "body": "Keep these lines.\n\nSuggested fix: restore the block.\n\n```python\nx = 1\n```",
    }
    findings = lint_review.ReviewLinter(
        {"body": body, "comments": [comment_missing_start_side]}, None
    ).run()
    assert any(
        f.level == "ERROR" and "found side='LEFT', start_side='RIGHT'" in f.message
        for f in findings
    )

    comment_with_start_side = dict(comment_missing_start_side, start_side="LEFT")
    assert (
        lint_review.ReviewLinter(
            {"body": body, "comments": [comment_with_start_side]}, None
        ).run()
        == []
    )


def test_linter_checks_diff_locations_and_existing_threads_with_pr(lint_review):
    fake_pr = SimpleNamespace(
        head_sha="b" * 40,
        diff_lines={
            ("rpent/x.py", "RIGHT"): {10: 0, 11: 0, 30: 1},
            ("rpent/x.py", "LEFT"): {10: 0},
        },
        commented_lines={("rpent/x.py", "RIGHT", 11): "alice"},
    )
    payload = {
        "commit_id": "a" * 40,
        "body": _valid_body(lint_review),
        "comments": [
            {
                "path": "rpent/x.py",
                "start_line": 10,
                "line": 30,
                "body": "Crosses hunks.\n\nSuggested fix: split.\n\n```python\nx = 1\n```",
            },
            {
                "path": "rpent/x.py",
                "line": 11,
                "body": "Existing thread.\n\nSuggested fix: reply.\n\n```python\nx = 2\n```",
            },
        ],
    }
    findings = lint_review.ReviewLinter(payload, fake_pr).run()
    assert any(
        "is not the PR head" in f.message for f in findings if f.level == "ERROR"
    )
    assert any(
        "span two diff hunks" in f.message for f in findings if f.level == "ERROR"
    )
    assert any(
        "@alice already has a thread" in f.message
        for f in findings
        if f.level == "WARN"
    )


def test_cli_template_lint_and_post(lint_review, monkeypatch, tmp_path, capsys):
    assert lint_review.main(["template"]) == 0
    template_out = capsys.readouterr().out
    assert "### 1. What settings did you use to reproduce this PR?" in template_out

    payload_file = tmp_path / "review.json"
    payload_file.write_text(
        json.dumps({"body": _valid_body(lint_review), "comments": []}),
        encoding="utf-8",
    )
    assert lint_review.main(["lint", str(payload_file)]) == 0

    posted: list[dict] = []

    class DummyPR:
        def __init__(self, repo: str, number: int) -> None:
            self.repo = repo
            self.number = number
            self.head_sha = "c" * 40
            self.diff_lines = {}
            self.commented_lines = {}

        def post(self, data: dict) -> str:
            posted.append(data)
            return "https://github.com/RLinf/RPent/pull/364#pullrequestreview-1"

    monkeypatch.setattr(lint_review, "PullRequest", DummyPR)
    assert lint_review.main(["post", "--pr", "364", str(payload_file)]) == 0
    assert len(posted) == 1
    assert posted[0]["commit_id"] == "c" * 40
    assert posted[0]["event"] == "COMMENT"

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

"""Capture API tasks, model calls, tool outcomes, and environment actions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import TypeAdapter
from pydantic_ai import ModelRetry, RunContext, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_core import to_jsonable_python

from rpent.session import EnvState, MissingInfo, Payload
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.planner.base import PlannerResult

logger = get_logger("task_capture")

SDK_VERSION = version("pydantic-ai-slim")
_REQUEST_PARAMETERS = TypeAdapter(ModelRequestParameters)
_SDK_VALUES = TypeAdapter(Any)


def start_api_task(
    state: EnvState, *, instruction: str, config: dict[str, Any]
) -> EnvState:
    """Open API capture with a serializable snapshot of the execution config."""
    state.start_task(
        instruction=instruction,
        config=Payload(
            format="rpent.task_config",
            format_version="1",
            data=to_jsonable_python(config),
            completeness="complete",
        ),
    )
    return state


def end_api_task(
    state: EnvState | None,
    result: PlannerResult | None,
    *,
    finish_result: dict | None,
    environment_success: bool | None,
    exception: BaseException | None,
    cancelled_reason: str | None = None,
) -> None:
    """Close capture after toolkit cleanup, keeping outcome separate from success."""
    if state is None:
        return
    error = result.error if result is not None else None
    reason = getattr(result, "stop_reason", None)
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"
        status = (
            "cancelled"
            if isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
        )
        reason = "operator" if status == "cancelled" else "execution_error"
    elif cancelled_reason is not None:
        status, reason = "cancelled", cancelled_reason
    elif (finish_result or {}).get("operator_aborted"):
        status, reason = "cancelled", "operator"
    elif (finish_result or {}).get("operator_finished"):
        status, reason = "completed", "operator_finished"
    elif reason == "cancelled":
        status = "cancelled"
    elif error is not None or result is None:
        status, reason = "failed", reason or "planner_error"
    else:
        status = "completed"
        reason = reason or ("finish" if finish_result is not None else "model_output")
    state.end_task(
        status=status,
        stop_reason=reason,
        result=Payload(
            format="rpent.task_result",
            format_version="1",
            data={
                "finish_result": finish_result,
                "environment_success": environment_success,
                "stats": result.stats if result is not None else {},
                "error": error,
            },
            completeness="complete",
        ),
    )
    try:
        path = state.save_planner_trajectory()
    except Exception:
        logger.exception("Failed to save planner trajectory")
    else:
        logger.info("Planner trajectory saved to %s", path)


class ToolCallCapture:
    """Bind execution updates to one registered tool call."""

    def __init__(self, state: EnvState, tool_call_id: str) -> None:
        self.state = state
        self.tool_call_id = tool_call_id
        self.started = False
        self.finished = False
        self._result_captured = False
        self.action_source: Literal["tool_computed", "direct"] = "tool_computed"
        self.prediction_id: str | None = None

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Propagate this call through async tasks and to_thread workers."""
        token = current_tool_call.set(self)
        try:
            yield
        finally:
            current_tool_call.reset(token)

    def start(
        self, *, action_source: Literal["tool_computed", "direct"] = "tool_computed"
    ) -> None:
        """Record actual admission to execution."""
        self.state.update_tool_call(
            self.tool_call_id, status="running", started_at=datetime.now(timezone.utc)
        )
        self.action_source = action_source
        self.started = True

    def result(self, payload: Payload) -> None:
        """Save the original result before any adapter transformation."""
        self.state.update_tool_call(
            self.tool_call_id, raw_result=payload, missing={"raw_result": None}
        )
        self._result_captured = True

    def finish(
        self, status: Literal["completed", "failed", "cancelled"], detail: str = ""
    ) -> None:
        """Finish execution without inventing timestamps for unstarted calls."""
        missing = {}
        if not self._result_captured:
            missing["raw_result"] = MissingInfo(
                "not_recorded" if self.started else "not_applicable", detail
            )
        self.state.update_tool_call(
            self.tool_call_id,
            status=status if self.started else "not_executed",
            ended_at=datetime.now(timezone.utc) if self.started else None,
            missing=missing,
        )
        self.finished = True


current_tool_call: ContextVar[ToolCallCapture | None] = ContextVar(
    "rpent_tool_call", default=None
)


class CapturingEnvClient:
    """Record executed environment actions for the current API tool call.

    During capture, ``capture_call(method, *args, **kwargs)`` executes the
    operation and returns the structure below. Without a callback, the wrapped
    client's method is called with ``capture=True`` and must return it directly::

        {
            "result": original_result,
            "before_observation": observation_payload,
            "steps": [
                {
                    "action": action_payload,
                    "after_observation": observation_payload,
                    "result": result_payload,
                }
            ],
        }

    Payloads use ``session.records.Payload`` fields, including ``format`` and
    ``data``. Steps contain only confirmed actions in execution order, with
    independent observation snapshots; each step's after-observation is the next
    step's before-observation. This client assigns record IDs, saves payloads,
    and returns ``result`` to the caller. Without an active capture for this
    state, calls go directly to the wrapped client.
    """

    def __init__(
        self,
        wrapped: Any,
        state: EnvState,
        *,
        capture_call: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.state = state
        self._capture_call = capture_call
        self._observation_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the client and discard the previous observation association."""
        self._observation_id = None
        return self.wrapped.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Execute and record one action, returning the original client result."""
        return self._execute("step", *args, **kwargs)

    def chunk_step(self, *args: Any, **kwargs: Any) -> Any:
        """Execute and record an action batch, returning the original client result."""
        return self._execute("chunk_step", *args, **kwargs)

    @contextmanager
    def prediction(self, instruction: str) -> Iterator[str | None]:
        """Group executed actions under one VLA prediction."""
        capture = current_tool_call.get()
        if capture is None or capture.state is not self.state:
            yield None
            return
        prediction_id = self.state.register_prediction(
            tool_call_id=capture.tool_call_id,
            instruction=instruction,
        )
        previous = capture.prediction_id
        capture.prediction_id = prediction_id
        try:
            yield prediction_id
        finally:
            capture.prediction_id = previous

    def _execute(
        self, method: Literal["step", "chunk_step"], *args: Any, **kwargs: Any
    ) -> Any:
        operation = getattr(self.wrapped, method)
        capture = current_tool_call.get()
        if capture is None or capture.state is not self.state:
            self._observation_id = None
            return operation(*args, **kwargs)
        try:
            response = (
                self._capture_call(method, *args, **kwargs)
                if self._capture_call is not None
                else operation(*args, capture=True, **kwargs)
            )
        except BaseException as exc:
            self._observation_id = None
            self.state.update_tool_call(
                capture.tool_call_id,
                missing={
                    "after_observation_id": MissingInfo(
                        "capture_failed",
                        f"Environment execution is unconfirmed: {type(exc).__name__}: {exc}",
                    )
                },
            )
            raise
        if not response["steps"]:
            return response["result"]

        before_id = self._observation_id or self.state.register_observation(
            payload=self._payload(response["before_observation"])
        )
        first_before_id = before_id
        for step in response["steps"]:
            after_id = self.state.register_observation(
                payload=self._payload(step["after_observation"])
            )
            self.state.register_action(
                tool_call_id=capture.tool_call_id,
                source="vla"
                if capture.prediction_id is not None
                else capture.action_source,
                prediction_id=capture.prediction_id,
                action=self._payload(step["action"]),
                before_observation_id=before_id,
                after_observation_id=after_id,
                result=self._payload(step["result"]),
            )
            before_id = after_id
        self._observation_id = before_id
        tool_call = self.state.get_tool_call(capture.tool_call_id)
        self.state.update_tool_call(
            capture.tool_call_id,
            before_observation_id=tool_call.before_observation_id or first_before_id,
            after_observation_id=before_id,
            missing={"before_observation_id": None, "after_observation_id": None},
        )
        return response["result"]

    def _payload(self, data: dict[str, Any]) -> Payload:
        """Decode a payload and save its non-inline arrays and bytes as attachments."""
        payload = Payload(**data)
        payload.missing = {
            path: MissingInfo(**reason) for path, reason in payload.missing.items()
        }

        def store(value: Any, pointer: str = "") -> Any:
            if isinstance(value, np.ndarray) and pointer in payload.inline_array_paths:
                return value
            if isinstance(value, (np.ndarray, bytes)):
                try:
                    return self.state.save_trajectory_artifact(value)
                except OSError as exc:
                    payload.completeness = "partial"
                    payload.missing[pointer] = MissingInfo(
                        "capture_failed", f"{type(exc).__name__}: {exc}"
                    )
                    return None
            if isinstance(value, dict):
                return {
                    key: store(
                        item, f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
                    )
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [
                    store(item, f"{pointer}/{index}")
                    for index, item in enumerate(value)
                ]
            return value

        payload.data = store(payload.data)
        return payload


def _sdk_data(value: Any) -> Any:
    """Normalize a Python-mode SDK snapshot while retaining binary attachments."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        return {key: _sdk_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sdk_data(item) for item in value]
    return to_jsonable_python(value, bytes_mode="base64")


def sdk_payload(format: str, data: Any) -> Payload:
    """Preserve SDK values, including binary content, in a serializable snapshot."""
    return Payload(
        format=format,
        format_version=SDK_VERSION,
        data=_sdk_data(_SDK_VALUES.dump_python(data, mode="python")),
        completeness="complete",
    )


class CapturingModel(WrapperModel):
    """Capture model requests, responses, and tool executions in one conversation."""

    def __init__(self, wrapped: Model, state: EnvState, conversation_id: str) -> None:
        super().__init__(wrapped)
        self.state = state
        self.conversation_id = conversation_id
        self.run_key: tuple[str | None, int] = (None, 0)
        self.calls: dict[tuple[str | None, int, str], ToolCallCapture] = {}

    def before_request(self, ctx: RunContext) -> None:
        self.run_key = (ctx.run_id, ctx.run_step)

    def response(self, planner_id: str, response: ModelResponse) -> None:
        """Register every issued call before SDK validation or dispatch."""
        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            execution_missing = MissingInfo(
                "not_applicable"
                if part.tool_name in {"read_image", "finish"}
                else "not_recorded",
                "This adapter operation uses no execution model or instruction."
                if part.tool_name in {"read_image", "finish"}
                else "Robot execution metadata is not captured.",
            )
            arguments = sdk_payload("pydantic_ai.tool_args", part.args)
            if response.state in {"incomplete", "interrupted"}:
                arguments.completeness = "partial"
                arguments.missing["completion"] = MissingInfo(
                    "not_recorded", "Model stream ended before the response completed."
                )
            tool_id = self.state.register_tool_call(
                planner_id=planner_id,
                name=part.tool_name,
                arguments=arguments,
                source_ids={"pydantic_ai.tool_call_id": part.tool_call_id},
                missing={
                    "raw_result": MissingInfo("pending", "Waiting for execution."),
                    "feedback": MissingInfo("pending", "Waiting for the adapter."),
                    "effective_instruction": execution_missing,
                    "model_config": execution_missing,
                    "before_observation_id": MissingInfo(
                        "not_recorded", "Observation capture is not connected."
                    ),
                    "after_observation_id": MissingInfo(
                        "not_recorded", "Observation capture is not connected."
                    ),
                },
            )
            self.calls[(*self.run_key, part.tool_call_id)] = ToolCallCapture(
                self.state, tool_id
            )

    @contextmanager
    def tool(self, ctx: RunContext) -> Iterator[ToolCallCapture]:
        """Capture adapter exceptions after physical workers have settled."""
        call = self.calls[(ctx.run_id, ctx.run_step, ctx.tool_call_id)]
        with call.bind():
            try:
                yield call
            except ModelRetry as exc:
                self.feedback(ctx, "pydantic_ai.model_retry", {"message": str(exc)})
                raise
            except BaseException:
                self.state.update_tool_call(
                    call.tool_call_id,
                    missing={
                        "feedback": MissingInfo(
                            "not_recorded", "Adapter did not return feedback."
                        )
                    },
                )
                raise
            finally:
                if not call.finished:
                    call.finish("failed", "Adapter exited before execution completed.")

    def feedback(self, ctx: RunContext, format: str, value: Any) -> None:
        call = self.calls[(ctx.run_id, ctx.run_step, ctx.tool_call_id)]
        self.state.update_tool_call(
            call.tool_call_id,
            feedback=sdk_payload(format, value),
            missing={"feedback": None},
        )

    def settle(self, run_id: str | None, detail: str, step: int | None = None) -> None:
        """Close undispatched calls and release settled capture handles."""
        for key, call in list(self.calls.items()):
            call_run, call_step, _ = key
            if call_run != run_id or (step is not None and call_step != step):
                continue
            if not call.finished:
                call.finish("cancelled", detail)
                self.state.update_tool_call(
                    call.tool_call_id,
                    missing={"feedback": MissingInfo("not_applicable", detail)},
                )
            del self.calls[key]

    def _start(
        self,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        parameters: ModelRequestParameters,
    ) -> str:
        source_ids = {"pydantic_ai.conversation_id": self.conversation_id}
        if self.run_key[0] is not None:
            source_ids["pydantic_ai.run_id"] = self.run_key[0]
        return self.state.register_planner(
            backend="api",
            request=Payload(
                format="pydantic_ai.model_request",
                format_version=SDK_VERSION,
                data={
                    "model": {"provider": self.system, "name": self.model_name},
                    "messages": _sdk_data(
                        ModelMessagesTypeAdapter.dump_python(messages, mode="python")
                    ),
                    "model_settings": to_jsonable_python(settings, bytes_mode="base64"),
                    "model_request_parameters": _REQUEST_PARAMETERS.dump_python(
                        parameters, mode="json"
                    ),
                },
                completeness="complete",
            ),
            source_ids=source_ids,
            started_at=datetime.now(timezone.utc),
            missing={"response": MissingInfo("pending", "Waiting for response.")},
        )

    def _end(
        self,
        planner_id: str,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        status = "completed"
        if error is not None:
            status = (
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed"
            )
        elif response is not None and response.state in {"incomplete", "interrupted"}:
            status = "cancelled" if response.state == "interrupted" else "failed"
        payload = None
        ids = {}
        missing = {"response": None}
        if response is not None:
            payload = Payload(
                format="pydantic_ai.model_response",
                format_version=SDK_VERSION,
                data=_sdk_data(
                    ModelMessagesTypeAdapter.dump_python([response], mode="python")[0]
                ),
                completeness="complete" if status == "completed" else "partial",
            )
            if status != "completed":
                payload.missing["completion"] = MissingInfo(
                    "not_recorded",
                    str(error)
                    if error is not None and str(error)
                    else "Response did not complete.",
                )
            if response.provider_response_id is not None:
                ids["provider.response_id"] = response.provider_response_id
        else:
            missing["response"] = MissingInfo(
                "capture_failed", f"{type(error).__name__}: {error}"
            )
        self.state.update_planner(
            planner_id,
            status=status,
            response=payload,
            source_ids=ids,
            ended_at=datetime.now(timezone.utc),
            missing=missing,
        )
        if response is not None:
            self.response(planner_id, response)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        planner_id = self._start(messages, model_settings, model_request_parameters)
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
        except BaseException as exc:
            self._end(planner_id, None, exc)
            raise
        self._end(planner_id, response, None)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        planner_id = self._start(messages, model_settings, model_request_parameters)
        stream = None
        error = None
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._end(planner_id, stream.get() if stream is not None else None, error)

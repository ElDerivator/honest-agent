"""Anthropic Messages API integration — gate a Claude tool-use agentic loop.

honest-agent does NOT depend on the ``anthropic`` SDK. The client is injected and
the response object is read structurally (``stop_reason`` + ``content`` blocks), so
this module imports with or without ``anthropic`` installed.

The standard manual agent loop (``client.messages.create`` → execute ``tool_use``
blocks → return ``tool_result`` blocks → repeat until ``end_turn``) produces no
honest verdict on its own: a loop that refuses, truncates at ``max_tokens``, or
spins to its iteration cap all *look* like they ended. This adapter gates that
loop on the environment.

Drive the loop and get a verdict (the common case — "bare Claude"):

    from anthropic import Anthropic
    from honest_agent.adapters.anthropic import gated_agent_loop

    client = Anthropic()

    def execute_tool(name, tool_input):
        if name == "run_tests":
            return "all 42 passed"
        return f"unknown tool: {name}"

    run = gated_agent_loop(
        client,
        messages=[{"role": "user", "content": "Run the tests and report."}],
        tools=TOOLS,
        execute_tool=execute_tool,
        vertical="ci",
        evidence=CommandExitsZero("pytest -q"),   # verifier — env decides ok
    )
    print(run.status)          # COMPLETED / UNVERIFIED / FAILED
    print(run.stop_reason)     # end_turn / refusal / max_tokens / max_iterations

Or wrap a loop you drive yourself:

    run = HonestAnthropicRun(vertical="ci")
    run.start()
    while True:
        resp = client.messages.create(...)
        run.observe(resp)                 # logs DARE, tracks the last stop_reason
        if resp.stop_reason == "end_turn":
            break
        ...                               # execute tools, append, continue
    run.verify("unit_test", "tests/")     # attach proof, or don't
    print(run.finish())                   # COMPLETED / UNVERIFIED / FAILED

Mapping from ``stop_reason`` to verdict:

    end_turn      -> COMPLETED with verifiable evidence, else UNVERIFIED
    refusal       -> FAILED (the model declined; the work did not happen)
    max_tokens    -> FAILED (response truncated mid-thought; not verifiably done)
    <iter cap>    -> FAILED (loop never converged)
    raised exc    -> FAILED, then re-raised (oracle is the environment)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from honest_agent.episode import close_episode, log_dare, open_episode
from honest_agent.event_log import EventLog

# A verifier is a zero-arg callable returning an evidence dict; a plain dict is
# hand-asserted evidence. Either is accepted wherever ``Evidence`` appears.
Evidence = Union[Dict[str, Any], Callable[[], Dict[str, Any]]]

# stop_reasons that mean the turn ended without the work being verifiably done.
_FAIL_STOP_REASONS = {"refusal", "max_tokens", "stop_sequence"}


def _content_blocks(response: Any) -> List[Any]:
    """The response's content list, tolerant of dicts or SDK objects."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    return list(content or [])


def _block_attr(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _stop_reason(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        return response.get("stop_reason")
    return getattr(response, "stop_reason", None)


class HonestAnthropicRun:
    """Turns one Claude agentic run into an evidence-gated episode.

    Wrap a loop you drive yourself: ``start()`` once, ``observe(response)`` each
    turn, attach proof with ``verify()``, then ``finish()`` (or ``fail()`` on an
    error you caught). The final ``stop_reason`` decides the verdict unless you
    override it.
    """

    def __init__(
        self,
        *,
        vertical: str,
        trigger: str = "anthropic_agent_run",
        allowed_verticals: Optional[Sequence[str]] = None,
        log: Optional[EventLog] = None,
    ) -> None:
        self.vertical = vertical
        self.trigger = trigger
        self.allowed_verticals = allowed_verticals
        self.log = log
        self._task_id: Optional[str] = None
        self._evidence: Optional[Dict[str, Any]] = None
        self._last_stop_reason: Optional[str] = None
        self._tool_calls = 0
        self.status: Optional[str] = None

    def start(self) -> str:
        """Open the episode. Returns the task id."""
        self._task_id = open_episode(
            vertical=self.vertical,
            trigger=self.trigger,
            allowed_verticals=self.allowed_verticals,
            log=self.log,
        )
        return self._task_id

    def verify(self, type: str, ref: str, ok: bool = True) -> None:
        """Attach machine-checkable proof for this run."""
        self._evidence = {"type": type, "ref": ref, "ok": ok}

    def observe(self, response: Any) -> List[Any]:
        """Record one turn. Logs the assistant's reasoning/tool calls as DARE and
        tracks the latest ``stop_reason``. Returns the turn's ``tool_use`` blocks
        so the caller can execute them."""
        if self._task_id is None:
            raise RuntimeError("call start() before observe()")
        self._last_stop_reason = _stop_reason(response)
        tool_uses = [b for b in _content_blocks(response) if _block_attr(b, "type") == "tool_use"]
        if tool_uses:
            self._tool_calls += len(tool_uses)
            log_dare(
                self._task_id, "A",
                {"tool_calls": [_block_attr(b, "name") for b in tool_uses]},
                log=self.log,
            )
        else:
            log_dare(self._task_id, "R", {"stop_reason": self._last_stop_reason}, log=self.log)
        return tool_uses

    def fail(self, error: BaseException) -> str:
        """Close the run FAILED on a caught exception. Returns the status."""
        if self._task_id is None:
            raise RuntimeError("call start() before fail()")
        log_dare(self._task_id, "E", {"error": repr(error)}, log=self.log)
        self.status = close_episode(self._task_id, "FAILED", log=self.log)
        self._task_id = None
        return self.status

    def finish(self, stop_reason: Optional[str] = None) -> str:
        """Close the run. The terminal ``stop_reason`` (the last observed one, or
        an explicit override) decides the verdict: a refusal/truncation is FAILED;
        a clean ``end_turn`` is COMPLETED-with-evidence, else UNVERIFIED."""
        if self._task_id is None:
            raise RuntimeError("call start() before finish()")
        reason = stop_reason or self._last_stop_reason
        if reason in _FAIL_STOP_REASONS:
            log_dare(self._task_id, "E", {"stop_reason": reason}, log=self.log)
            self.status = close_episode(self._task_id, "FAILED", log=self.log)
        else:
            self.status = close_episode(
                self._task_id, "COMPLETED", evidence=self._evidence, log=self.log,
            )
        self._task_id = None
        return self.status


@dataclass
class AgentRun:
    """The honest verdict on a gated Claude loop."""
    status: str                       # COMPLETED / UNVERIFIED / FAILED
    stop_reason: Optional[str]        # end_turn / refusal / max_tokens / max_iterations
    iterations: int
    tool_calls: int
    final_message: Any                # the last Anthropic response object, or None


def gated_agent_loop(
    client: Any,
    *,
    messages: List[Dict[str, Any]],
    execute_tool: Callable[[str, Any], Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-opus-4-8",
    max_tokens: int = 16000,
    vertical: str = "agent",
    trigger: str = "anthropic_agent_loop",
    allowed_verticals: Optional[Sequence[str]] = None,
    evidence: Optional[Evidence] = None,
    max_iterations: int = 50,
    log: Optional[EventLog] = None,
    **create_kwargs: Any,
) -> AgentRun:
    """Drive the standard Claude tool-use loop and return an honest verdict.

    Runs ``client.messages.create`` until Claude stops calling tools, executing
    each ``tool_use`` block via ``execute_tool(name, input)`` and feeding all
    ``tool_result`` blocks back in a single user turn (parallel-tool-use safe).
    ``pause_turn`` is resumed transparently. A propagated exception from the API
    call or from ``execute_tool`` closes the episode FAILED and re-raises — the
    oracle of truth is the environment, not the model's claim that it finished.

    ``evidence`` is a dict or a verifier callable evaluated at close; without
    verifiable evidence a clean ``end_turn`` is recorded UNVERIFIED, never as a
    positive signal.
    """
    run = HonestAnthropicRun(
        vertical=vertical, trigger=trigger,
        allowed_verticals=allowed_verticals, log=log,
    )
    task_id = run.start()
    convo = list(messages)
    iterations = 0
    last_response: Any = None

    try:
        log_dare(task_id, "D", {"model": model, "n_tools": len(tools or [])}, log=log)
        while True:
            if iterations >= max_iterations:
                log_dare(task_id, "E", {"reason": "max_iterations", "limit": max_iterations}, log=log)
                status = close_episode(task_id, "FAILED", log=log)
                run._task_id = None
                return AgentRun(status, "max_iterations", iterations, run._tool_calls, last_response)
            iterations += 1

            request: Dict[str, Any] = {
                "model": model, "max_tokens": max_tokens, "messages": convo, **create_kwargs,
            }
            if tools:
                request["tools"] = tools
            response = client.messages.create(**request)
            last_response = response
            tool_uses = run.observe(response)
            reason = _stop_reason(response)

            if reason == "pause_turn":
                # Server-side tool hit its iteration limit; re-send to continue.
                convo = convo + [{"role": "assistant", "content": _content_blocks(response)}]
                continue

            if reason == "tool_use" or tool_uses:
                convo.append({"role": "assistant", "content": _content_blocks(response)})
                tool_results: List[Dict[str, Any]] = []
                for block in tool_uses:
                    result = execute_tool(_block_attr(block, "name"), _block_attr(block, "input"))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": _block_attr(block, "id"),
                        "content": result if isinstance(result, (list, str)) else str(result),
                    })
                convo.append({"role": "user", "content": tool_results})
                continue

            # Terminal stop_reason. A refusal/truncation is FAILED; a clean
            # end_turn is COMPLETED only with verifiable evidence, else UNVERIFIED.
            if reason in _FAIL_STOP_REASONS:
                log_dare(task_id, "E", {"stop_reason": reason}, log=log)
                status = close_episode(task_id, "FAILED", log=log)
            else:
                status = close_episode(task_id, "COMPLETED", evidence=evidence, log=log)
            run._task_id = None
            run.status = status
            return AgentRun(status, reason, iterations, run._tool_calls, last_response)
    except BaseException as exc:  # noqa: BLE001 — record FAILED, then propagate
        if run._task_id is not None:
            run.fail(exc)
        raise

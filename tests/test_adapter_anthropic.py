"""Unit-tests the Anthropic adapter by driving the documented Messages-API loop
with a fake client — no anthropic install, no network. The adapter reads the
response structurally (stop_reason + content blocks), so SimpleNamespace stand-ins
exercise exactly the same code paths as the real SDK objects."""
import os
import tempfile
from types import SimpleNamespace

import pytest

from honest_agent import EventLog
from honest_agent.adapters.anthropic import HonestAnthropicRun, gated_agent_loop


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def _text(stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text="done")])


def _tool_use(name="run_tests", tool_input=None, block_id="toolu_1"):
    block = SimpleNamespace(type="tool_use", name=name, input=tool_input or {}, id=block_id)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


class _FakeClient:
    """Returns scripted responses in order; records the tool_result turns it received."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):  # mimics client.messages.create
        self.calls += 1
        return self._scripted.pop(0)


def _run(scripted, *, execute_tool=None, **kw):
    return gated_agent_loop(
        _FakeClient(scripted),
        messages=[{"role": "user", "content": "go"}],
        execute_tool=execute_tool or (lambda n, i: "ok"),
        log=_log(),
        vertical="t",
        **kw,
    )


def test_clean_end_turn_without_evidence_is_unverified():
    run = _run([_text("end_turn")])
    assert run.status == "UNVERIFIED" and run.stop_reason == "end_turn"


def test_clean_end_turn_with_passing_verifier_completes():
    run = _run([_text("end_turn")], evidence=lambda: {"type": "unit_test", "ref": "tests/", "ok": True})
    assert run.status == "COMPLETED"


def test_failing_verifier_downgrades_to_unverified():
    run = _run([_text("end_turn")], evidence={"type": "unit_test", "ref": "tests/", "ok": False})
    assert run.status == "UNVERIFIED"


def test_tool_use_then_end_turn_executes_and_completes():
    seen = []
    run = _run(
        [_tool_use("run_tests", {"path": "tests/"}), _text("end_turn")],
        execute_tool=lambda name, inp: seen.append((name, inp)) or "all passed",
        evidence={"type": "unit_test", "ref": "tests/", "ok": True},
    )
    assert run.status == "COMPLETED" and run.tool_calls == 1
    assert seen == [("run_tests", {"path": "tests/"})]


def test_refusal_is_failed():
    run = _run([_text("refusal")])
    assert run.status == "FAILED" and run.stop_reason == "refusal"


def test_max_tokens_truncation_is_failed():
    run = _run([_text("max_tokens")])
    assert run.status == "FAILED" and run.stop_reason == "max_tokens"


def test_tool_executor_exception_is_failed_and_reraises():
    def boom(name, inp):
        raise RuntimeError("tool crashed")

    with pytest.raises(RuntimeError):
        _run([_tool_use()], execute_tool=boom)


def test_runaway_loop_hits_iteration_cap_and_fails():
    # The fake never stops calling tools; the cap must terminate it FAILED.
    run = _run([_tool_use() for _ in range(10)], max_iterations=3)
    assert run.status == "FAILED" and run.stop_reason == "max_iterations" and run.iterations == 3


def test_manual_run_observe_verify_finish():
    run = HonestAnthropicRun(vertical="t", log=_log())
    run.start()
    run.observe(_tool_use())          # a turn that called a tool
    run.observe(_text("end_turn"))    # clean finish
    run.verify("file_change", "out.json")
    assert run.finish() == "COMPLETED"


def test_manual_run_refusal_finishes_failed():
    run = HonestAnthropicRun(vertical="t", log=_log())
    run.start()
    run.observe(_text("refusal"))
    assert run.finish() == "FAILED"

"""Unit-tests the callback handler by driving the documented LangChain call
sequence directly — no langchain install required to test the adapter logic."""
import os
import tempfile

from honest_agent import EventLog
from honest_agent.adapters.langchain import HonestCallbackHandler


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def _handler():
    return HonestCallbackHandler(vertical="t", log=_log())


def test_run_without_evidence_is_unverified():
    h = _handler()
    h.on_chain_start({}, {})
    h.on_chain_end({"answer": 42})
    assert h.status == "UNVERIFIED"


def test_run_with_evidence_completes():
    h = _handler()
    h.on_chain_start({}, {})
    h.verify("unit_test", "tests/")
    h.on_chain_end({"answer": 42})
    assert h.status == "COMPLETED"


def test_errored_run_is_failed():
    h = _handler()
    h.on_chain_start({}, {})
    h.on_chain_error(RuntimeError("tool timeout"))
    assert h.status == "FAILED"


def test_nested_chains_collapse_to_one_episode():
    h = _handler()
    h.on_chain_start({}, {})      # outer
    h.on_chain_start({}, {})      # inner
    h.on_chain_end({})            # inner ends -> not closed yet
    assert h.status is None
    h.on_chain_end({})            # outer ends -> verdict now
    assert h.status == "UNVERIFIED"


def test_error_ignores_unwinding_callbacks():
    h = _handler()
    h.on_chain_start({}, {})
    h.on_chain_start({}, {})
    h.on_chain_error(RuntimeError("boom"))   # collapses the whole run
    h.on_chain_end({})                       # unwinding parent -> ignored
    assert h.status == "FAILED"

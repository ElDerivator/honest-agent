import os
import tempfile

import pytest

from honest_agent import EventLog, gated, honest_episode


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def test_context_manager_no_evidence_is_unverified():
    log = _log()
    with honest_episode(vertical="t", trigger="x", log=log) as outcome:
        pass
    assert outcome.status == "UNVERIFIED"


def test_context_manager_with_evidence_completes():
    log = _log()
    with honest_episode(vertical="t", trigger="x", log=log) as outcome:
        outcome.verify("unit_test", "tests/")
    assert outcome.status == "COMPLETED"


def test_context_manager_exception_is_failed_and_reraises():
    log = _log()
    with pytest.raises(RuntimeError):
        with honest_episode(vertical="t", trigger="x", log=log) as outcome:
            raise RuntimeError("boom")
    assert outcome.status == "FAILED"
    assert any(e["event_type"] == "dare_error" for e in log.read_all())


def test_decorator_returns_value_and_status():
    log = _log()

    @gated(vertical="t", trigger="x", log=log)
    def work(ok: bool):
        return {"type": "file_change", "ref": "a.py", "ok": True} if ok else None

    ev, status = work(True)
    assert status == "COMPLETED" and ev["ref"] == "a.py"
    _, status2 = work(False)
    assert status2 == "UNVERIFIED"


def test_decorator_exception_is_failed():
    log = _log()

    @gated(vertical="t", trigger="x", log=log)
    def work():
        raise ValueError("nope")

    result, status = work()
    assert result is None and status == "FAILED"

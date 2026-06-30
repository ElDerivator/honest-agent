import os
import tempfile

import pytest

from honest_agent import EventLog, close_episode, log_dare, open_episode


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def test_completed_without_evidence_downgrades_to_unverified():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    assert close_episode(t, "COMPLETED", log=log) == "UNVERIFIED"


def test_completed_with_evidence_holds():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    ev = {"type": "file_change", "ref": "a.py", "ok": True}
    assert close_episode(t, "COMPLETED", evidence=ev, log=log) == "COMPLETED"


def test_evidence_ok_must_be_true_not_truthy():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    assert close_episode(t, "COMPLETED", evidence={"ok": "yes"}, log=log) == "UNVERIFIED"


def test_failed_is_recorded_as_failed():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    assert close_episode(t, "FAILED", log=log) == "FAILED"


def test_invalid_phase_raises():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    with pytest.raises(ValueError):
        log_dare(t, "Z", {}, log=log)


def test_invalid_status_raises():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    with pytest.raises(ValueError):
        close_episode(t, "DONE", log=log)


def test_allowed_verticals_enforced():
    log = _log()
    with pytest.raises(ValueError):
        open_episode(vertical="evil", trigger="x", allowed_verticals=("ok",), log=log)


def test_events_are_appended_in_order():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    log_dare(t, "A", {"k": 1}, log=log)
    close_episode(t, "FAILED", log=log)
    events = log.read_all()
    assert [e["event_type"] for e in events] == [
        "episode_open", "dare_action", "episode_close",
    ]

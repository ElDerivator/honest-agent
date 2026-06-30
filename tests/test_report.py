import os
import tempfile

from honest_agent import EventLog, close_episode, open_episode
from honest_agent.report import report


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def test_report_counts_and_corruption_rate():
    log = _log()
    # one proven completion
    t = open_episode(vertical="t", trigger="x", log=log)
    close_episode(t, "COMPLETED", evidence={"type": "x", "ref": "y", "ok": True}, log=log)
    # three unverified claims
    for _ in range(3):
        t = open_episode(vertical="t", trigger="x", log=log)
        close_episode(t, "COMPLETED", log=log)
    # one failure
    t = open_episode(vertical="t", trigger="x", log=log)
    close_episode(t, "FAILED", log=log)

    r = report(log)
    assert (r.completed, r.unverified, r.failed) == (1, 3, 1)
    assert r.claimed_complete == 4
    assert abs(r.corruption_rate - 0.75) < 1e-9


def test_empty_log_is_zero_rate():
    assert report(_log()).corruption_rate == 0.0


def test_unknown_status_is_counted_not_dropped():
    # Real-world logs carry statuses we don't recognize (e.g. "EMPTY") — surface, never drop.
    log = _log()
    log.append({"event_type": "episode_close", "episode_status": "EMPTY"})
    log.append({"event_type": "episode_close", "episode_status": "WEIRD"})
    assert report(log).other == 2

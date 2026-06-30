import json
import os
import tempfile

from honest_agent import EventLog, verify_chain


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def _rewrite(path: str, lines: list) -> None:
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_intact_chain_verifies():
    log = _log()
    for i in range(5):
        log.append({"event_type": "episode_close", "i": i})
    check = log.verify_chain()
    assert check.ok and check.length == 5


def test_content_tamper_is_detected():
    log = _log()
    for i in range(5):
        log.append({"event_type": "episode_close", "status": "UNVERIFIED", "i": i})
    lines = open(log.path).read().splitlines()
    rec = json.loads(lines[2])
    rec["status"] = "COMPLETED"          # the classic lie: flip the verdict
    lines[2] = json.dumps(rec)
    _rewrite(log.path, lines)
    check = verify_chain(log.path)
    assert not check.ok and check.broken_at == 2 and check.reason == "content tampered"


def test_deletion_is_detected():
    log = _log()
    for i in range(5):
        log.append({"event_type": "x", "i": i})
    lines = open(log.path).read().splitlines()
    del lines[2]
    _rewrite(log.path, lines)
    check = verify_chain(log.path)
    assert not check.ok and check.broken_at == 2


def test_chain_continues_across_instances():
    log = _log()
    path = log.path
    log.append({"event_type": "a"})
    EventLog(path).append({"event_type": "b"})   # fresh instance loads the tip
    assert EventLog(path).verify_chain().ok

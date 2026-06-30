import os
import tempfile

from honest_agent import EventLog, close_episode, open_episode
from honest_agent.verifiers import (
    AllOf,
    ArtifactPresent,
    CommandExitsZero,
    FileExists,
    MetricThreshold,
)


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def _tmpfile(content: str = "x") -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    with open(path, "w") as fh:
        fh.write(content)
    return path


def test_file_exists_true_and_false():
    path = _tmpfile()
    assert FileExists(path)()["ok"] is True
    assert FileExists(path + ".nope")()["ok"] is False


def test_command_exit_codes():
    assert CommandExitsZero("exit 0")()["ok"] is True
    assert CommandExitsZero("exit 1")()["ok"] is False


def test_metric_threshold():
    assert MetricThreshold(0.95, ">=", 0.9)()["ok"] is True
    assert MetricThreshold(0.80, ">=", 0.9)()["ok"] is False


def test_all_of_is_and():
    path = _tmpfile()
    assert AllOf(FileExists(path), MetricThreshold(1.0, "==", 1.0))()["ok"] is True
    assert AllOf(FileExists(path), MetricThreshold(0.0, "==", 1.0))()["ok"] is False


def test_honor_system_hole_closed_by_verifier():
    # The exact lie from the audit: claim a test file that does not exist.
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    status = close_episode(
        t, "COMPLETED", evidence=FileExists("tests/que_no_existe.py"), log=log,
    )
    assert status == "UNVERIFIED"   # the library checked, and caught it


def test_real_passing_check_completes():
    log = _log()
    t = open_episode(vertical="t", trigger="x", log=log)
    assert close_episode(t, "COMPLETED", evidence=ArtifactPresent(_tmpfile()), log=log) == "COMPLETED"

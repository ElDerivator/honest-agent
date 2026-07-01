import json
import os
import subprocess
import tempfile

import pytest

from honest_agent import (
    EventLog,
    PreconditionsFailed,
    close_episode,
    open_episode,
)
from honest_agent.verifiers import (
    AllOf,
    ArtifactPresent,
    CommandExitsZero,
    Corroborate,
    FileExists,
    GitCommitPresent,
    GitDiffNonEmpty,
    GitTreeClean,
    MetricThreshold,
    ReceiptVerified,
)

_HAS_GIT = subprocess.run(  # noqa: S603,S607 — test-time capability probe
    ["git", "--version"], capture_output=True
).returncode == 0
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="git not installed")


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


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return proc.stdout


def _git_repo() -> str:
    repo = tempfile.mkdtemp()
    _git(repo, "init", "-q")
    with open(os.path.join(repo, "a.txt"), "w") as fh:
        fh.write("first\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "seed commit")
    return repo


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


# --- git evidence -----------------------------------------------------------

@_needs_git
def test_git_commit_present():
    repo = _git_repo()
    assert GitCommitPresent("HEAD", cwd=repo)()["ok"] is True
    assert GitCommitPresent("does-not-exist", cwd=repo)()["ok"] is False


@_needs_git
def test_git_commit_message_contains():
    repo = _git_repo()
    assert GitCommitPresent("HEAD", contains="seed", cwd=repo)()["ok"] is True
    assert GitCommitPresent("HEAD", contains="nope", cwd=repo)()["ok"] is False


@_needs_git
def test_git_commit_present_no_repo_is_false():
    empty = tempfile.mkdtemp()
    assert GitCommitPresent("HEAD", cwd=empty)()["ok"] is False


@_needs_git
def test_git_tree_clean_true_then_dirty():
    repo = _git_repo()
    assert GitTreeClean(cwd=repo)()["ok"] is True
    with open(os.path.join(repo, "a.txt"), "a") as fh:
        fh.write("uncommitted\n")
    ev = GitTreeClean(cwd=repo)()
    assert ev["ok"] is False
    assert ev["dirty"]                     # names the dirty path


@_needs_git
def test_git_diff_non_empty():
    repo = _git_repo()
    with open(os.path.join(repo, "a.txt"), "w") as fh:
        fh.write("second\n")
    _git(repo, "commit", "-q", "-am", "change")
    assert GitDiffNonEmpty("HEAD~1", "HEAD", cwd=repo)()["ok"] is True
    # same ref against itself: no diff -> not proof of work
    assert GitDiffNonEmpty("HEAD", "HEAD", cwd=repo)()["ok"] is False
    # scoped to an untouched path -> no diff
    assert GitDiffNonEmpty("HEAD~1", "HEAD", paths=["nope.txt"], cwd=repo)()["ok"] is False


# --- receipt evidence -------------------------------------------------------

def test_receipt_verified_match_and_mismatch():
    path = _tmpfile(json.dumps({"status": "ok", "worker": {"exit": 0}}))
    assert ReceiptVerified(path, "status", "ok")()["ok"] is True
    assert ReceiptVerified(path, "status", "fail")()["ok"] is False
    assert ReceiptVerified(path, "worker.exit", 0)()["ok"] is True      # dot-path
    assert ReceiptVerified(path, "worker.missing", 0)()["ok"] is False


def test_receipt_missing_file_is_false():
    ev = ReceiptVerified("/no/such/receipt.json")()
    assert ev["ok"] is False and "error" in ev


# --- corroboration ----------------------------------------------------------

def test_corroborate_needs_min_sources_to_agree():
    ok1, ok2 = FileExists(_tmpfile()), MetricThreshold(1.0, "==", 1.0)
    bad = MetricThreshold(0.0, "==", 1.0)
    assert Corroborate(ok1, ok2, bad, min_sources=2)()["ok"] is True    # 2 of 3
    assert Corroborate(ok1, bad, bad, min_sources=2)()["ok"] is False   # 1 of 3
    assert Corroborate(ok1, bad, min_sources=2)()["n_ok"] == 1


def test_corroborate_rejects_too_few_verifiers():
    with pytest.raises(ValueError):
        Corroborate(FileExists(_tmpfile()), min_sources=2)


# --- pre-dispatch gate ------------------------------------------------------

def test_preconditions_block_open_and_log():
    log = _log()
    with pytest.raises(PreconditionsFailed) as exc:
        open_episode(vertical="deploy", trigger="ci",
                     preconditions=FileExists("/no/such/precondition"), log=log)
    assert exc.value.evidence["ok"] is False
    events = log.read_all()
    assert len(events) == 1
    assert events[0]["event_type"] == "episode_blocked"
    assert events[0]["episode_status"] == "BLOCKED"


def test_preconditions_pass_opens_normally():
    log = _log()
    t = open_episode(vertical="deploy", trigger="ci",
                     preconditions=FileExists(_tmpfile()), log=log)
    assert t.startswith("ep-deploy-")
    assert log.read_all()[0]["event_type"] == "episode_open"


def test_no_preconditions_is_unchanged():
    log = _log()
    t = open_episode(vertical="deploy", trigger="ci", log=log)
    assert log.read_all()[0]["event_type"] == "episode_open"
    assert t

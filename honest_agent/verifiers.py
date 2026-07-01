"""Verifiers — let honest-agent compute `ok` from the environment instead of
trusting a hand-written `ok: True`. A verifier is a zero-arg callable returning
an evidence dict with a *computed* `ok`. Pass one where evidence is expected and
the library runs the check itself:

    from honest_agent import close_episode
    from honest_agent.verifiers import CommandExitsZero

    close_episode(task, "COMPLETED", evidence=CommandExitsZero("pytest -q"))
    # honest-agent actually runs pytest and sets ok from the exit code.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

Evidence = Dict[str, Any]


def _git(args: Sequence[str], cwd: Optional[str] = None,
         timeout: float = 30.0) -> tuple:
    """Run ``git <args>``; return ``(returncode, stdout, stderr)``. A missing git
    or a non-repo directory is reported as a non-zero code, never an exception —
    the verifier's job is to answer ``ok``, not to blow up the close."""
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except (FileNotFoundError, OSError) as exc:
        return 127, "", repr(exc)


class Verifier:
    """Base: a zero-arg callable that checks the environment and returns evidence."""

    type = "verifier"

    def __call__(self) -> Evidence:  # pragma: no cover - abstract
        raise NotImplementedError


class FileExists(Verifier):
    type = "file_exists"

    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self) -> Evidence:
        return {"type": self.type, "ref": self.path, "ok": os.path.isfile(self.path)}


class ArtifactPresent(Verifier):
    """A file that exists AND is at least ``min_bytes`` long."""

    type = "artifact_present"

    def __init__(self, path: str, min_bytes: int = 1) -> None:
        self.path = path
        self.min_bytes = min_bytes

    def __call__(self) -> Evidence:
        size = os.path.getsize(self.path) if os.path.isfile(self.path) else 0
        return {"type": self.type, "ref": self.path,
                "ok": size >= self.min_bytes and os.path.isfile(self.path), "bytes": size}


class CommandExitsZero(Verifier):
    """Run a command; ok iff it exits 0. The real test, actually executed."""

    type = "command"

    def __init__(self, cmd: Union[str, Sequence[str]], timeout: float = 120.0,
                 cwd: Optional[str] = None) -> None:
        self.cmd = cmd
        self.timeout = timeout
        self.cwd = cwd

    def __call__(self) -> Evidence:
        ref = self.cmd if isinstance(self.cmd, str) else " ".join(self.cmd)
        try:
            proc = subprocess.run(
                self.cmd, shell=isinstance(self.cmd, str),
                capture_output=True, timeout=self.timeout, cwd=self.cwd,
            )
            return {"type": self.type, "ref": ref,
                    "ok": proc.returncode == 0, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"type": self.type, "ref": ref, "ok": False, "error": "timeout"}
        except (FileNotFoundError, OSError) as exc:
            return {"type": self.type, "ref": ref, "ok": False, "error": repr(exc)}


_OPS: Dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


class MetricThreshold(Verifier):
    """ok iff ``value <op> threshold`` (e.g. accuracy >= 0.9)."""

    type = "metric_threshold"

    def __init__(self, value: float, op: str, threshold: float, name: str = "metric") -> None:
        if op not in _OPS:
            raise ValueError(f"unsupported op {op!r}; use one of {sorted(_OPS)}")
        self.value, self.op, self.threshold, self.name = value, op, threshold, name

    def __call__(self) -> Evidence:
        return {"type": self.type, "ok": _OPS[self.op](self.value, self.threshold),
                "ref": f"{self.name} {self.value} {self.op} {self.threshold}",
                "value": self.value, "threshold": self.threshold}


class AllOf(Verifier):
    """AND of several verifiers. ok iff every sub-verifier is ok."""

    type = "all_of"

    def __init__(self, *verifiers: Callable[[], Evidence]) -> None:
        if not verifiers:
            raise ValueError("AllOf requires at least one verifier")
        self.verifiers = verifiers

    def __call__(self) -> Evidence:
        results: List[Evidence] = [v() for v in self.verifiers]
        return {"type": self.type, "ref": f"{len(results)} checks",
                "ok": all(r.get("ok") is True for r in results), "checks": results}


class GitCommitPresent(Verifier):
    """ok iff ``rev`` resolves to a real commit — and, if ``contains`` is given,
    its message contains that substring. Proof the agent's "committed it" is a
    commit that exists, not a branch name it typed."""

    type = "git_commit"

    def __init__(self, rev: str = "HEAD", contains: Optional[str] = None,
                 cwd: Optional[str] = None) -> None:
        self.rev = rev
        self.contains = contains
        self.cwd = cwd

    def __call__(self) -> Evidence:
        code, out, _ = _git(["rev-parse", "--verify", f"{self.rev}^{{commit}}"], self.cwd)
        sha = out.strip() if code == 0 else None
        ok = sha is not None
        if ok and self.contains is not None:
            mcode, msg, _ = _git(["log", "-1", "--format=%B", sha], self.cwd)  # type: ignore[list-item]
            ok = mcode == 0 and self.contains in msg
        return {"type": self.type, "ref": self.rev, "ok": ok, "sha": sha}


class GitTreeClean(Verifier):
    """ok iff the working tree has no uncommitted changes (``git status
    --porcelain`` is empty). A clean tree means the work landed in a commit — not
    that it's still dangling, unstaged, about to be lost."""

    type = "git_tree_clean"

    def __init__(self, cwd: Optional[str] = None) -> None:
        self.cwd = cwd

    def __call__(self) -> Evidence:
        code, out, err = _git(["status", "--porcelain"], self.cwd)
        dirty = out.strip().splitlines()
        ev: Evidence = {"type": self.type, "ref": self.cwd or ".",
                        "ok": code == 0 and not dirty, "dirty": dirty}
        if code != 0:
            ev["error"] = err.strip() or f"git exit {code}"
        return ev


class GitDiffNonEmpty(Verifier):
    """ok iff there is a real diff between ``base`` and ``ref`` (optionally limited
    to ``paths``). Proof the work *changed files*, not that a commit or branch
    merely exists. Uses ``git diff --quiet``: exit 1 means differences."""

    type = "git_diff"

    def __init__(self, base: str, ref: str = "HEAD",
                 paths: Optional[Sequence[str]] = None, cwd: Optional[str] = None) -> None:
        self.base = base
        self.ref = ref
        self.paths = list(paths) if paths else None
        self.cwd = cwd

    def __call__(self) -> Evidence:
        args = ["diff", "--quiet", self.base, self.ref]
        if self.paths:
            args += ["--", *self.paths]
        code, _, err = _git(args, self.cwd)
        ev: Evidence = {"type": self.type, "ref": f"{self.base}..{self.ref}", "ok": code == 1}
        if code not in (0, 1):
            ev["ok"] = False
            ev["error"] = err.strip() or f"git exit {code}"
        return ev


class ReceiptVerified(Verifier):
    """Read a JSON receipt file; ok iff the value at dot-path ``key`` equals
    ``expected``. The ``receipt`` evidence type made concrete — a dispatch/worker
    receipt is only proof if it says what you claim it says."""

    type = "receipt"

    def __init__(self, path: str, key: str = "status", expected: Any = "ok") -> None:
        self.path = path
        self.key = key
        self.expected = expected

    def __call__(self) -> Evidence:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data: Any = json.load(fh)
        except (OSError, ValueError) as exc:
            return {"type": self.type, "ref": self.path, "ok": False, "error": repr(exc)}
        value: Any = data
        for part in self.key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        return {"type": self.type, "ref": f"{self.path}#{self.key}",
                "ok": value == self.expected, "value": value}


class Corroborate(Verifier):
    """N-of-M independent witnesses. ok iff at least ``min_sources`` sub-verifiers
    are ok. ``AllOf`` demands every check; this demands agreement among several
    independent ones — one source can be gamed, three that concur is harder to
    fake (git says committed, the file is there, the receipt confirms)."""

    type = "corroborate"

    def __init__(self, *verifiers: Callable[[], Evidence], min_sources: int = 2) -> None:
        if min_sources < 1:
            raise ValueError("min_sources must be >= 1")
        if len(verifiers) < min_sources:
            raise ValueError(
                f"Corroborate needs at least min_sources ({min_sources}) verifiers, "
                f"got {len(verifiers)}"
            )
        self.verifiers = verifiers
        self.min_sources = min_sources

    def __call__(self) -> Evidence:
        results: List[Evidence] = [v() for v in self.verifiers]
        n_ok = sum(1 for r in results if r.get("ok") is True)
        return {"type": self.type,
                "ref": f"{n_ok}/{len(results)} sources agree, need {self.min_sources}",
                "ok": n_ok >= self.min_sources, "n_ok": n_ok, "checks": results}

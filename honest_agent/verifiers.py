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

import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

Evidence = Dict[str, Any]


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

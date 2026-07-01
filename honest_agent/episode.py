"""Evidence-gated episode lifecycle for autonomous agents.

An episode is one unit of work with a `task_id` and an explicit lifecycle:

    open_episode()  ->  log_dare() * N  ->  close_episode()

Core rule — honesty as architecture: an episode only closes ``COMPLETED`` when
there is evidence of real state (a test that passed, a file that changed, a
verified receipt). A ``COMPLETED`` claim without verifiable evidence is
downgraded to ``UNVERIFIED`` and never counts as a positive signal. The oracle
of truth is the environment, not the model's opinion.

Every event is appended to an immutable log — no update, no delete.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional, Sequence, Union

from honest_agent.event_log import EventLog, default_log

# D.A.R.E. phases -> canonical event_type.
_DARE = {
    "D": "dare_decision",    # the reasoning / chain-of-thought before acting
    "A": "dare_action",      # the executed command + immediate result
    "R": "dare_reflection",  # self-evaluation: did it go as expected?
    "E": "dare_error",       # exception / stack trace (gold for contrast)
}
_TERMINAL = {"COMPLETED", "FAILED"}

# Evidence is a computed dict or a verifier callable that produces one. The same
# abstraction gates both ends: preconditions on open, proof on close.
Evidence = Union[Dict[str, Any], Callable[[], Dict[str, Any]]]


class PreconditionsFailed(RuntimeError):
    """Raised by :func:`open_episode` when its ``preconditions`` verifier is not
    ok — the environment isn't in a state where the work can honestly begin. The
    would-be ``task_id`` and the failing evidence are attached; the block is also
    recorded to the log as an ``episode_blocked`` event, so a refused dispatch is
    counted, not invisible."""

    def __init__(self, task_id: str, evidence: Dict[str, Any]) -> None:
        self.task_id = task_id
        self.evidence = evidence
        super().__init__(f"preconditions not met for {task_id}: {evidence.get('ref')}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_task_id(vertical: str) -> str:
    """Human-readable episode id: ``ep-<vertical>-<uuid8>``."""
    return f"ep-{vertical}-{uuid.uuid4().hex[:8]}"


def open_episode(
    *,
    vertical: str,
    trigger: str,
    task_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    allowed_verticals: Optional[Sequence[str]] = None,
    preconditions: Optional[Evidence] = None,
    log: Optional[EventLog] = None,
) -> str:
    """Open an episode (state RUNNING). Returns the task_id (generated if absent).

    If ``allowed_verticals`` is given, ``vertical`` must be a member or the call
    is rejected — a scope guard, not a default restriction.

    A pre-dispatch gate symmetric with the close gate: if ``preconditions`` (a
    dict or verifier, same shape as ``close_episode``'s ``evidence``) is given and
    does not yield ``ok is True``, the episode is not opened — the block is logged
    as ``episode_blocked`` and :class:`PreconditionsFailed` is raised. Don't start
    work the environment says can't succeed; the oracle gates the entry, not just
    the exit.
    """
    if not vertical or not trigger:
        raise ValueError("open_episode requires vertical and trigger")
    if allowed_verticals is not None and vertical not in allowed_verticals:
        raise ValueError(f"vertical {vertical!r} not in allowed_verticals")
    log = log or default_log()
    task_id = task_id or new_task_id(vertical)

    if preconditions is not None:
        checked = preconditions() if callable(preconditions) else preconditions
        if not (checked and checked.get("ok") is True):
            log.append({
                "ts": _now(),
                "event_type": "episode_blocked",
                "task_id": task_id,
                "vertical": vertical,
                "episode_status": "BLOCKED",
                "trigger": trigger,
                "evidence": checked or {},
                "payload": meta or {},
            })
            raise PreconditionsFailed(task_id, checked or {})

    log.append({
        "ts": _now(),
        "event_type": "episode_open",
        "task_id": task_id,
        "vertical": vertical,
        "episode_status": "RUNNING",
        "trigger": trigger,
        "payload": meta or {},
    })
    return task_id


def log_dare(
    task_id: str,
    phase: str,
    payload: Any,
    *,
    vertical: Optional[str] = None,
    log: Optional[EventLog] = None,
) -> bool:
    """Record a D.A.R.E. event bound to an episode. ``phase`` is D | A | R | E."""
    if not task_id:
        raise ValueError("log_dare requires task_id")
    key = (phase or "").strip().upper()
    if key not in _DARE:
        raise ValueError(f"invalid phase: {phase!r} (use D/A/R/E)")
    log = log or default_log()
    return log.append({
        "ts": _now(),
        "event_type": _DARE[key],
        "task_id": task_id,
        "vertical": vertical,
        "dare_phase": key,
        "payload": payload,
    })


def close_episode(
    task_id: str,
    status: str,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    log: Optional[EventLog] = None,
) -> str:
    """Close an episode with a terminal status. Returns the *resolved* status.

    Honesty gate: closing ``COMPLETED`` requires ``evidence`` with ``ok is True``.
    ``evidence = {"type": "unit_test|file_change|receipt|human_approval",
    "ref": "<id/path>", "ok": bool}``. Without verifiable evidence a success is
    downgraded to ``UNVERIFIED`` — it is recorded, but never as a positive signal.
    """
    if not task_id:
        raise ValueError("close_episode requires task_id")
    status = (status or "").strip().upper()
    if status not in _TERMINAL:
        raise ValueError(f"invalid status: {status!r} (use COMPLETED/FAILED)")

    if callable(evidence):
        evidence = evidence()       # a verifier: let the environment decide ok
    verified = bool(evidence and evidence.get("ok") is True)
    if status == "COMPLETED" and not verified:
        status = "UNVERIFIED"

    log = log or default_log()
    log.append({
        "ts": _now(),
        "event_type": "episode_close",
        "task_id": task_id,
        "episode_status": status,
        "verified": verified,
        "evidence": evidence or {},
        "payload": {},
    })
    return status

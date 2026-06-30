"""Ergonomic surface: wrap any agent's task boundary with an evidence gate.

Two forms, same rule — work isn't COMPLETED unless it's proven:

    with honest_episode(vertical="deploy", trigger="ci") as outcome:
        run_the_thing()
        outcome.verify("unit_test", "tests/")     # attach proof, or don't
    print(outcome.status)                          # COMPLETED / UNVERIFIED / FAILED

    @gated(vertical="deploy", trigger="ci")
    def deploy():
        ...
        return {"type": "unit_test", "ref": "tests/", "ok": True}   # evidence, or None
"""
from __future__ import annotations

import functools
import inspect
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

from honest_agent.episode import close_episode, log_dare, open_episode
from honest_agent.event_log import EventLog


class Outcome:
    """Handle the wrapped block uses to attach proof. No proof -> UNVERIFIED."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.status: Optional[str] = None
        self.evidence: Optional[Dict[str, Any]] = None

    def verify(self, type: str, ref: str) -> "Outcome":
        """Attach hand-asserted evidence (use sparingly — prefer .check())."""
        self.evidence = {"type": type, "ref": ref, "ok": True}
        return self

    def check(self, verifier: Callable[[], Dict[str, Any]]) -> "Outcome":
        """Run a verifier now and store its *computed* evidence (real env check)."""
        self.evidence = verifier()
        return self


@contextmanager
def honest_episode(
    *,
    vertical: str,
    trigger: str,
    allowed_verticals: Optional[Sequence[str]] = None,
    log: Optional[EventLog] = None,
) -> Iterator[Outcome]:
    """Wrap a block of work. Exit clean -> COMPLETED (downgraded to UNVERIFIED
    without evidence). Raise -> the exception is logged and the episode FAILED."""
    task_id = open_episode(
        vertical=vertical, trigger=trigger,
        allowed_verticals=allowed_verticals, log=log,
    )
    outcome = Outcome(task_id)
    try:
        yield outcome
    except Exception as exc:
        log_dare(task_id, "E", {"error": repr(exc)}, log=log)
        outcome.status = close_episode(task_id, "FAILED", log=log)
        raise
    else:
        outcome.status = close_episode(
            task_id, "COMPLETED", evidence=outcome.evidence, log=log,
        )


@asynccontextmanager
async def honest_episode_async(
    *,
    vertical: str,
    trigger: str,
    allowed_verticals: Optional[Sequence[str]] = None,
    log: Optional[EventLog] = None,
):
    """Async form of honest_episode — drop it inside async agents/graphs."""
    task_id = open_episode(
        vertical=vertical, trigger=trigger,
        allowed_verticals=allowed_verticals, log=log,
    )
    outcome = Outcome(task_id)
    try:
        yield outcome
    except Exception as exc:
        log_dare(task_id, "E", {"error": repr(exc)}, log=log)
        outcome.status = close_episode(task_id, "FAILED", log=log)
        raise
    else:
        outcome.status = close_episode(
            task_id, "COMPLETED", evidence=outcome.evidence, log=log,
        )


def gated(
    *,
    vertical: str,
    trigger: str,
    allowed_verticals: Optional[Sequence[str]] = None,
    log: Optional[EventLog] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Tuple[Any, str]]]:
    """Decorate a function whose return value is its evidence dict (or None).

    Returns ``(return_value, status)``. Works on both sync and ``async def``
    functions — a coroutine is wrapped in an awaitable returning the same tuple."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Tuple[Any, str]]:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Tuple[Any, str]:
                task_id = open_episode(
                    vertical=vertical, trigger=trigger,
                    allowed_verticals=allowed_verticals, log=log,
                )
                try:
                    evidence = await fn(*args, **kwargs)
                except Exception as exc:
                    log_dare(task_id, "E", {"error": repr(exc)}, log=log)
                    return None, close_episode(task_id, "FAILED", log=log)
                return evidence, close_episode(task_id, "COMPLETED", evidence=evidence, log=log)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, str]:
            task_id = open_episode(
                vertical=vertical, trigger=trigger,
                allowed_verticals=allowed_verticals, log=log,
            )
            try:
                evidence = fn(*args, **kwargs)
            except Exception as exc:
                log_dare(task_id, "E", {"error": repr(exc)}, log=log)
                return None, close_episode(task_id, "FAILED", log=log)
            return evidence, close_episode(task_id, "COMPLETED", evidence=evidence, log=log)
        return wrapper
    return deco

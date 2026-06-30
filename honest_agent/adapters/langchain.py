"""LangChain / LangGraph integration via a callback handler.

Implements langchain-core's ``BaseCallbackHandler`` contract. honest-agent does
NOT depend on langchain — the base is imported softly, so this module loads with
or without it. Install ``langchain-core`` to use the handler in a real graph:

    from honest_agent.adapters.langchain import HonestCallbackHandler

    handler = HonestCallbackHandler(vertical="research")
    graph.invoke(state, config={"callbacks": [handler]})
    # ... after the run:
    print(handler.status)        # COMPLETED / UNVERIFIED / FAILED

Every top-level chain/agent run becomes an episode. A run that ends without
verifiable evidence is recorded COMPLETED-but-UNVERIFIED; an errored run is
FAILED. Attach proof from inside a node with ``handler.verify(type, ref)``.

Nested chains are collapsed into the outermost run (depth-tracked), so the graph
produces one honest verdict per top-level invocation, not one per sub-chain.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:  # real base when langchain-core is installed
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - optional dependency absent
    class BaseCallbackHandler:  # minimal shim; module stays importable
        pass

from honest_agent.episode import close_episode, log_dare, open_episode
from honest_agent.event_log import EventLog


class HonestCallbackHandler(BaseCallbackHandler):
    """Turns each top-level LangChain/LangGraph run into an evidence-gated episode."""

    def __init__(
        self,
        *,
        vertical: str,
        trigger: str = "langchain_run",
        allowed_verticals: Optional[Sequence[str]] = None,
        log: Optional[EventLog] = None,
    ) -> None:
        self.vertical = vertical
        self.trigger = trigger
        self.allowed_verticals = allowed_verticals
        self.log = log
        self._task_id: Optional[str] = None
        self._evidence: Optional[Dict[str, Any]] = None
        self._depth = 0
        self.status: Optional[str] = None

    def verify(self, type: str, ref: str) -> None:
        """Attach machine-checkable proof for the current run."""
        self._evidence = {"type": type, "ref": ref, "ok": True}

    # --- BaseCallbackHandler contract -------------------------------------
    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        if self._depth == 0:
            self._evidence = None
            self.status = None
            self._task_id = open_episode(
                vertical=self.vertical, trigger=self.trigger,
                allowed_verticals=self.allowed_verticals, log=self.log,
            )
        self._depth += 1

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0 and self._task_id is not None:
            try:
                marker = list(outputs)[:20] if hasattr(outputs, "__iter__") else str(type(outputs))
            except Exception:
                marker = "<unreadable>"
            log_dare(self._task_id, "A", {"output_keys": marker}, log=self.log)
            self.status = close_episode(
                self._task_id, "COMPLETED", evidence=self._evidence, log=self.log,
            )
            self._task_id = None

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        if self._task_id is None:
            return
        log_dare(self._task_id, "E", {"error": repr(error)}, log=self.log)
        self.status = close_episode(self._task_id, "FAILED", log=self.log)
        self._task_id = None
        self._depth = 0  # collapse the failed run; ignore the unwinding callbacks

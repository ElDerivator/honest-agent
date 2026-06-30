"""Append-only, tamper-evident event log.

Each record is hash-chained: it stores the sha256 of its own canonical content
plus the previous record's hash. Editing, inserting, or deleting any record
breaks the chain, and `verify_chain()` reports exactly where. The record of what
an agent claimed cannot be quietly rewritten after the fact.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

GENESIS = "0" * 64
_CHAIN_FIELDS = ("_prev", "_hash")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_record(core: Dict[str, Any], prev: str) -> str:
    return hashlib.sha256((_canonical(core) + prev).encode("utf-8")).hexdigest()


@dataclass
class ChainCheck:
    ok: bool
    length: int
    broken_at: Optional[int] = None
    reason: str = "intact"

    def __str__(self) -> str:
        if self.ok:
            return f"chain intact: {self.length} records"
        return f"chain BROKEN at record {self.broken_at}: {self.reason} ({self.length} records total)"


class EventLog:
    """Thread-safe append-only JSONL log with hash-chained tamper evidence."""

    def __init__(self, path: str = "honest_agent_events.jsonl") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._tip: Optional[str] = None

    def _load_tip(self) -> str:
        if self._tip is not None:
            return self._tip
        last = None
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                for ln in fh:
                    if ln.strip():
                        last = ln
        if last:
            try:
                self._tip = json.loads(last).get("_hash", GENESIS)
            except Exception:
                self._tip = GENESIS
        else:
            self._tip = GENESIS
        return self._tip

    def append(self, event: Dict[str, Any]) -> bool:
        with self._lock:
            prev = self._load_tip()
            core = {k: v for k, v in event.items() if k not in _CHAIN_FIELDS}
            digest = _hash_record(core, prev)
            record = {**core, "_prev": prev, "_hash": digest}
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(record) + "\n")
            self._tip = digest
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def verify_chain(self) -> ChainCheck:
        prev = GENESIS
        records = self.read_all()
        for i, rec in enumerate(records):
            stored_hash = rec.get("_hash")
            stored_prev = rec.get("_prev")
            if stored_hash is None or stored_prev is None:
                return ChainCheck(False, len(records), i, "unchained record (no hash)")
            if stored_prev != prev:
                return ChainCheck(False, len(records), i, "broken link — record inserted or deleted")
            core = {k: v for k, v in rec.items() if k not in _CHAIN_FIELDS}
            if stored_hash != _hash_record(core, stored_prev):
                return ChainCheck(False, len(records), i, "content tampered")
            prev = stored_hash
        return ChainCheck(True, len(records))


_DEFAULT: Optional[EventLog] = None


def default_log() -> EventLog:
    """Process-wide default log (writes to ./honest_agent_events.jsonl)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = EventLog()
    return _DEFAULT


def verify_chain(source: Union[EventLog, str]) -> ChainCheck:
    """Verify the hash chain of a log (or its path). Reports the first break."""
    log = source if isinstance(source, EventLog) else EventLog(source)
    return log.verify_chain()

"""Quantify the lie. Read an episode log and report how many "successes" were
actually unverified.

    python -m honest_agent.report honest_agent_events.jsonl
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Union

from honest_agent.event_log import EventLog


@dataclass
class Report:
    completed: int = 0
    unverified: int = 0
    failed: int = 0
    other: int = 0   # episode_close events with an unrecognized status — never dropped silently

    @property
    def claimed_complete(self) -> int:
        """Times the agent said "done" — proven or not."""
        return self.completed + self.unverified

    @property
    def corruption_rate(self) -> float:
        """Share of "done" claims with no verifiable evidence (0.0 if none claimed)."""
        return self.unverified / self.claimed_complete if self.claimed_complete else 0.0

    def __str__(self) -> str:
        return (
            "honest-agent report\n"
            f"  completed (proven):   {self.completed}\n"
            f"  unverified (claimed): {self.unverified}\n"
            f"  failed:               {self.failed}\n"
            f"  other (unrecognized): {self.other}\n"
            f"  corruption rate:      {self.corruption_rate:.1%} "
            f"({self.unverified}/{self.claimed_complete} 'done' claims had no proof)"
        )


def report(source: Union[EventLog, str]) -> Report:
    """Aggregate episode_close events from a log (or its path) into a Report."""
    log = source if isinstance(source, EventLog) else EventLog(source)
    r = Report()
    for ev in log.read_all():
        if ev.get("event_type") != "episode_close":
            continue
        status = ev.get("episode_status")
        if status == "COMPLETED":
            r.completed += 1
        elif status == "UNVERIFIED":
            r.unverified += 1
        elif status == "FAILED":
            r.failed += 1
        else:
            r.other += 1
    return r


def main(argv: list = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    path = args[0] if args else "honest_agent_events.jsonl"
    print(report(path))


if __name__ == "__main__":
    main()

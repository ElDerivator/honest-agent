"""Quantify the lie. Read an episode log and report how many "successes" were
actually unproven — including the ones that bypassed the gate entirely.

    python -m honest_agent.report honest_agent_events.jsonl

Two fingerprints of a bypassed gate, both surfaced here:

* **off-contract status** — an ``episode_close`` whose ``episode_status`` is none
  of COMPLETED/UNVERIFIED/FAILED (e.g. a hand-rolled emitter writing "EMPTY").
  ``close_episode`` would reject it; a raw ``log.append`` does not.
* **ungated completion** — ``episode_status == "COMPLETED"`` without
  ``verified is True``. The gate downgrades any unverified COMPLETED to
  UNVERIFIED, so a surviving COMPLETED that isn't ``verified`` never went through
  ``close_episode``: a "done" nothing blessed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Union

from honest_agent.event_log import EventLog


@dataclass
class Report:
    completed: int = 0           # episode_status == COMPLETED (gate-verified + ungated)
    unverified: int = 0
    failed: int = 0
    other: int = 0               # status outside the contract — bypassed close_episode
    ungated_complete: int = 0    # COMPLETED without verified is True — the gate never blessed it

    @property
    def gate_verified(self) -> int:
        """COMPLETED records the gate actually blessed (``verified is True``)."""
        return self.completed - self.ungated_complete

    @property
    def claimed_complete(self) -> int:
        """Times the agent said "done" — proven or not."""
        return self.completed + self.unverified

    @property
    def unproven(self) -> int:
        """"done" claims with no gate-blessed proof: unverified + ungated completions."""
        return self.unverified + self.ungated_complete

    @property
    def corruption_rate(self) -> float:
        """Share of "done" claims the gate never proved (0.0 if none claimed).

        Counts both kinds of unproven completion: those honestly recorded
        UNVERIFIED, and those that bypassed the gate to assert COMPLETED.
        """
        return self.unproven / self.claimed_complete if self.claimed_complete else 0.0

    def __str__(self) -> str:
        return (
            "honest-agent report\n"
            f"  completed (gate-verified): {self.gate_verified}\n"
            f"  completed (ungated):       {self.ungated_complete}   <- claimed COMPLETED, gate never verified it\n"
            f"  unverified (claimed):      {self.unverified}\n"
            f"  failed:                    {self.failed}\n"
            f"  off-contract status:       {self.other}   <- status outside the contract; bypassed close_episode\n"
            f"  corruption rate:           {self.corruption_rate:.1%} "
            f"({self.unproven}/{self.claimed_complete} 'done' claims unproven)"
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
            if ev.get("verified") is not True:
                r.ungated_complete += 1
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

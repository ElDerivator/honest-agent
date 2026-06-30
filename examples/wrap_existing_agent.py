"""Drop the gate around an agent you already have. It catches the lie for you.

Run:  PYTHONPATH=.. python wrap_existing_agent.py
"""
from honest_agent import EventLog, honest_episode

log = EventLog("/tmp/honest_agent_wrap_demo.jsonl")


def agent_step(claim_only: bool):
    """Pretend this is your existing agent doing real work."""
    if claim_only:
        return None                      # it "did" the work but can't prove it
    return ("unit_test", "tests/")       # it produced checkable proof


# Case A — the agent finishes and just says "trust me".
with honest_episode(vertical="deploy", trigger="ci", log=log) as outcome:
    proof = agent_step(claim_only=True)
    if proof:
        outcome.verify(*proof)
print(f"[claim only]  -> {outcome.status}")

# Case B — same flow, but the agent attaches a passing test.
with honest_episode(vertical="deploy", trigger="ci", log=log) as outcome:
    proof = agent_step(claim_only=False)
    if proof:
        outcome.verify(*proof)
print(f"[with proof]  -> {outcome.status}")

# Case C — the agent blows up mid-task. No silent success.
try:
    with honest_episode(vertical="deploy", trigger="ci", log=log) as outcome:
        raise RuntimeError("tool call timed out")
except RuntimeError:
    pass
print(f"[exception]   -> {outcome.status}")

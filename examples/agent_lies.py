"""The hook: an agent claims success. Does it hold?

Run:  PYTHONPATH=.. python agent_lies.py
"""
from honest_agent import EventLog, close_episode, log_dare, open_episode

log = EventLog("/tmp/honest_agent_demo.jsonl")

# Scenario 1 — the agent says it's done, with nothing to back it up.
t1 = open_episode(vertical="demo", trigger="user_request", log=log)
log_dare(t1, "A", {"cmd": "deploy", "result": "looks fine to me"}, log=log)
status1 = close_episode(t1, "COMPLETED", log=log)
print(f"[no evidence]   agent claimed COMPLETED  ->  recorded as: {status1}")

# Scenario 2 — same claim, backed by a passing test.
t2 = open_episode(vertical="demo", trigger="user_request", log=log)
log_dare(t2, "A", {"cmd": "pytest", "result": "12 passed"}, log=log)
status2 = close_episode(
    t2, "COMPLETED",
    evidence={"type": "unit_test", "ref": "tests/", "ok": True},
    log=log,
)
print(f"[with evidence] agent claimed COMPLETED  ->  recorded as: {status2}")

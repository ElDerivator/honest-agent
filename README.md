# honest-agent

**Evidence-gated task completion for autonomous agents.** An agent can't mark work
`COMPLETED` unless it can prove it — otherwise the claim is recorded as
`UNVERIFIED`. The oracle of truth is the environment, not the model.

## Why this exists

Spend enough months building agent systems and the same failure keeps surfacing:
the agent says it's done. The deploy "succeeded." The task is "complete." Then you
look, and the artifact isn't there, the test never ran, or the metric contradicts
the claim. Outcome-only logging credits it as a win. It wasn't one.

The lesson that took those months to land is plain: **a success an agent can't
prove isn't a success.** Most stacks treat completion as something the model
asserts. honest-agent treats it as something the environment has to confirm — a
passing test, a changed file, a verified receipt — or the claim is downgraded and
never counts as a positive signal.

It's not a new idea so much as an unshipped one. The research backs the pain:
procedure-aware evaluation finds [27–78% of agent "successes" are procedurally
corrupt](https://arxiv.org/abs/2603.03116), and evidence-bound governance
[drives false claims to zero where verification-only pipelines leave them near
25%](https://arxiv.org/abs/2511.05524). Those are papers and eval frameworks.
honest-agent is the small runtime primitive you wrap around your own agent.

## What it is

~150 lines, standard-library only, framework-agnostic. It is **not** a memory
framework or an agent platform — it sits on top of whatever you already run
(LangChain/LangGraph, CrewAI, your own loop) and adds one thing: an honest verdict
on whether the work actually happened.

## Install

```bash
pip install honest-agent             # core, zero dependencies
pip install "honest-agent[langchain]"  # + the LangChain/LangGraph adapter
```

## Three ways to use it

**1 — The episode lifecycle** (explicit control):

```python
from honest_agent import open_episode, close_episode

task = open_episode(vertical="deploy", trigger="ci")
close_episode(task, "COMPLETED")                     # no proof  -> "UNVERIFIED"
close_episode(task, "COMPLETED",
              evidence={"type": "unit_test", "ref": "tests/", "ok": True})  # -> "COMPLETED"
```

**2 — The context manager** (wrap a block):

```python
from honest_agent import honest_episode

with honest_episode(vertical="deploy", trigger="ci") as outcome:
    run_the_thing()
    outcome.verify("unit_test", "tests/")    # attach proof, or don't
print(outcome.status)                        # COMPLETED / UNVERIFIED / FAILED
```

For async agents, use `honest_episode_async` (same API) or put `@gated` on an
`async def` — both run inside an event loop:

```python
async with honest_episode_async(vertical="deploy", trigger="ci") as outcome:
    await run_the_thing()
    outcome.verify("unit_test", "tests/")
```

**3 — The LangChain / LangGraph adapter** (drop it on a graph you already have):

```python
from honest_agent.adapters.langchain import HonestCallbackHandler

handler = HonestCallbackHandler(vertical="research")
graph.invoke(state, config={"callbacks": [handler]})
print(handler.status)
```

## The rule, exactly

Closing `COMPLETED` requires `evidence` with `ok is True`
(`{"type": "unit_test|file_change|receipt|human_approval", "ref": "...", "ok": bool}`).
Without it, the status is recorded `UNVERIFIED` — logged, but never counted as a
positive signal. A raised exception is recorded `FAILED`. The event log is
append-only: nothing is updated or deleted.

## Verify for real, don't take its word

`evidence={"ok": True}` is hand-asserted — honest-agent will believe it. The point
is to *not* take the agent's word: pass a **verifier** and the library computes
`ok` from the environment itself.

```python
from honest_agent import close_episode
from honest_agent.verifiers import CommandExitsZero, ArtifactPresent, AllOf

close_episode(task, "COMPLETED",
              evidence=AllOf(CommandExitsZero("pytest -q"),     # actually runs it
                             ArtifactPresent("dist/app.whl")))   # actually checks it
# ok is True only if pytest exits 0 AND the artifact exists. Otherwise -> UNVERIFIED.
```

Built-in verifiers: `FileExists`, `ArtifactPresent`, `CommandExitsZero`,
`MetricThreshold`, `AllOf`. A verifier is just a zero-arg callable returning an
evidence dict — write your own.

## Report the corruption rate

The gate prevents the lie; the report quantifies it. How many of your agent's
"done" claims had no proof?

```bash
honest-agent-report honest_agent_events.jsonl    # or: python -m honest_agent.report ...
```
```
honest-agent report
  completed (proven):   12
  unverified (claimed): 8
  failed:               3
  corruption rate:      40.0% (8/20 'done' claims had no proof)
```

## Tamper-evident by default

The log is hash-chained — each record carries the sha256 of its content plus the
previous record's hash. Edit, insert, or delete any line and the chain breaks;
`verify_chain` reports exactly where. The record of what your agent claimed can't
be quietly rewritten after the fact.

```python
from honest_agent import verify_chain

print(verify_chain("honest_agent_events.jsonl"))
# chain intact: 1240 records
# — or —  chain BROKEN at record 312: content tampered
```

## What it doesn't do

It doesn't decide *what* counts as evidence — you do. It doesn't run your agent,
hold its memory, or judge output quality. It does one thing: it stops "done" from
being something the model can simply assert.

## License

Apache-2.0.

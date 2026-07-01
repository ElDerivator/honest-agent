"""honest-agent — evidence-gated task completion for autonomous agents.

An agent can't mark work COMPLETED unless it can prove it. Otherwise the claim
is recorded as UNVERIFIED. The oracle of truth is the environment, not the model.
"""
from honest_agent.episode import (
    PreconditionsFailed,
    close_episode,
    log_dare,
    new_task_id,
    open_episode,
)
from honest_agent.event_log import ChainCheck, EventLog, default_log, verify_chain
from honest_agent.gated import Outcome, gated, honest_episode, honest_episode_async
from honest_agent.verifiers import (
    AllOf,
    ArtifactPresent,
    CommandExitsZero,
    Corroborate,
    FileExists,
    GitCommitPresent,
    GitDiffNonEmpty,
    GitTreeClean,
    MetricThreshold,
    ReceiptVerified,
    Verifier,
)

__all__ = [
    "open_episode",
    "log_dare",
    "close_episode",
    "new_task_id",
    "PreconditionsFailed",
    "EventLog",
    "default_log",
    "verify_chain",
    "ChainCheck",
    "honest_episode",
    "honest_episode_async",
    "gated",
    "Outcome",
    "Verifier",
    "FileExists",
    "ArtifactPresent",
    "CommandExitsZero",
    "MetricThreshold",
    "AllOf",
    "GitCommitPresent",
    "GitTreeClean",
    "GitDiffNonEmpty",
    "ReceiptVerified",
    "Corroborate",
]
__version__ = "0.4.0"

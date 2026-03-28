"""
models.py — Pydantic dataclass models for the SQL Query RL Environment.

Three models map directly to the OpenEnv spec:
  • SQLAction      → what the agent submits each step
  • SQLObservation → what the environment returns after each step
  • SQLState       → episode-level metadata tracked by the server
"""

from __future__ import annotations

from typing import Any, Literal, Optional


from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class SQLAction(Action):
    """The agent's action for a single step: a SQL query to execute."""

    fixed_query: str
    """The fixed SQL query string the agent is submitting for evaluation."""

    query: Optional[str] = None
    """Optional field to satisfy the OpenEnv base Action requirement."""

    explanation: Optional[str] = None
    """Optional human-readable description of what was changed in this query."""

    optimization_notes: Optional[str] = None
    """Optional notes about performance or structural optimizations applied."""


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SQLObservation(Observation):
    """Everything the environment tells the agent after executing its query."""

    done: bool
    """True when the episode has ended (success, failure, or step limit hit)."""

    reward: float
    """Scalar reward signal for the step."""

    original_query: str
    """The original broken / starter query given at the beginning of the episode."""

    last_query: str
    """The agent's most-recently submitted query (mirrors SQLAction.query)."""

    execution_result: str
    """Raw execution output on success, or the full error message on failure."""

    success: bool
    """True if the submitted query executed without raising a database error."""

    execution_time_ms: float
    """Wall-clock time taken to execute the query, in milliseconds."""

    result_rows: list[dict[str, Any]]
    """Rows returned by the query, each row represented as a column→value dict."""

    feedback: str
    """Natural-language explanation of what is right or wrong with the query."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SQLState(State):
    """Episode-level metadata maintained by the environment server."""

    episode_id: str
    """Unique identifier for the current episode."""

    step_count: int
    """Number of steps taken so far in the current episode."""

    task_name: str
    """Human-readable name of the current SQL repair / optimisation task."""

    difficulty: Literal["easy", "medium", "hard"]
    """Difficulty tier of the current task."""

    schema: str
    """The database schema (DDL) in use for this episode."""

    max_steps: int
    """Maximum number of steps allowed before the episode is forcibly ended."""

    current_score: float
    """Cumulative score accumulated by the agent during the episode so far."""

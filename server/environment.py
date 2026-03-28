"""
server/environment.py — Core RL environment for the SQL Query repair tasks.

SQLEnvironment inherits from the OpenEnv Environment base class and wires
together SQLite execution, reward computation, and episode state management.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server import Environment

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SQLAction
from models import SQLObservation, SQLState
from rewards import compute_reward

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_SCHEMAS_PATH = _DATA_DIR / "schemas.json"
_QUERIES_PATH = _DATA_DIR / "queries.json"

# ---------------------------------------------------------------------------
# Deterministic sample data for each schema
# ---------------------------------------------------------------------------

_SAMPLE_DATA: dict[str, dict[str, list[tuple[Any, ...]]]] = {
    "ecommerce": {
        "users": [
            (1, "Alice Johnson",  "alice@example.com",  "2023-01-10 08:00:00"),
            (2, "Bob Smith",      "bob@example.com",    "2023-02-14 09:30:00"),
            (3, "Carol White",    "carol@example.com",  "2023-03-05 11:15:00"),
            (4, "David Brown",    "david@example.com",  "2023-04-20 14:00:00"),
            (5, "Eva Martinez",   "eva@example.com",    "2023-05-18 16:45:00"),
        ],
        "orders": [
            (1, 1, 129.99, "shipped",    "2023-06-01 10:00:00"),
            (2, 1, 59.49,  "delivered",  "2023-06-15 12:00:00"),
            (3, 2, 249.00, "processing", "2023-07-02 09:00:00"),
            (4, 3, 34.95,  "cancelled",  "2023-07-10 15:30:00"),
            (5, 4, 99.00,  "shipped",    "2023-08-05 08:45:00"),
        ],
        "order_items": [
            (1, 1, 101, 2, 49.99),
            (2, 1, 102, 1, 29.99),
            (3, 2, 103, 3, 19.83),
            (4, 3, 104, 1, 249.00),
            (5, 4, 105, 5,  6.99),
        ],
    },
    "hr": {
        "employees": [
            (1, "Grace Lee",      1, 72000.00, "2019-03-15"),
            (2, "Henry Kim",      2, 85000.00, "2018-07-22"),
            (3, "Irene Patel",    1, 68000.00, "2020-01-10"),
            (4, "James Oliver",   3, 91000.00, "2017-11-05"),
            (5, "Karen Scott",    2, 77500.00, "2021-06-30"),
        ],
        "departments": [
            (1, "Engineering",   500000.00),
            (2, "Sales",         300000.00),
            (3, "Management",    750000.00),
            (4, "HR",            200000.00),
            (5, "Marketing",     250000.00),
        ],
        "attendance": [
            (1, 1, "2024-01-15", 8.0),
            (2, 2, "2024-01-15", 7.5),
            (3, 3, "2024-01-15", 8.0),
            (4, 4, "2024-01-15", 9.0),
            (5, 5, "2024-01-15", 6.5),
        ],
    },
    "blog": {
        "authors": [
            (1, "Lena Hart",    "lena@blog.com"),
            (2, "Marco Diaz",   "marco@blog.com"),
            (3, "Nina Zhao",    "nina@blog.com"),
            (4, "Oscar Tran",   "oscar@blog.com"),
            (5, "Paula Evans",  "paula@blog.com"),
        ],
        "posts": [
            (1, "Intro to SQL",          "Content A", 1, "2024-01-05 10:00:00", 1500),
            (2, "Advanced Joins",        "Content B", 2, "2024-02-10 11:00:00", 3200),
            (3, "Indexing Deep Dive",    "Content C", 1, "2024-03-01 09:00:00", 4100),
            (4, "Window Functions",      "Content D", 3, "2024-03-20 14:00:00", 2750),
            (5, "Query Optimisation",    "Content E", 2, "2024-04-15 08:30:00", 5300),
        ],
        "comments": [
            (1, 1, 2, "Great intro!",        "2024-01-06 12:00:00"),
            (2, 1, 3, "Very helpful.",        "2024-01-07 09:30:00"),
            (3, 2, 1, "Loved the examples.",  "2024-02-11 10:00:00"),
            (4, 3, 4, "Mind-blowing!",        "2024-03-02 11:15:00"),
            (5, 5, 5, "Best post ever.",      "2024-04-16 07:45:00"),
        ],
    },
}

# DDL templates derived from schemas.json column lists
_DDL: dict[str, dict[str, str]] = {
    "ecommerce": {
        "users":       "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, created_at TEXT)",
        "orders":      "CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL, status TEXT, created_at TEXT)",
        "order_items": "CREATE TABLE IF NOT EXISTS order_items (id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, price REAL)",
    },
    "hr": {
        "employees":   "CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, salary REAL, hire_date TEXT)",
        "departments": "CREATE TABLE IF NOT EXISTS departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL)",
        "attendance":  "CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, employee_id INTEGER, date TEXT, hours_worked REAL)",
    },
    "blog": {
        "authors":  "CREATE TABLE IF NOT EXISTS authors (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
        "posts":    "CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, title TEXT, content TEXT, author_id INTEGER, published_at TEXT, view_count INTEGER)",
        "comments": "CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY, post_id INTEGER, author_id INTEGER, body TEXT, created_at TEXT)",
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLEnvironment(Environment):
    """
    OpenEnv environment that challenges an RL agent to repair broken SQL queries.

    Episode lifecycle
    -----------------
    1. reset(task_name)  — load a task, set up the SQLite DB, return initial obs
    2. step(action)      — execute the submitted query, compute reward, return obs
    3. state (property)  — episode metadata for logging / the OpenEnv server
    """

    def __init__(self) -> None:
        super().__init__()

        # Load task catalogue
        with open(_SCHEMAS_PATH) as f:
            self._schemas: dict = json.load(f)
        with open(_QUERIES_PATH) as f:
            self._queries: list[dict] = json.load(f)
        self._query_index: dict[str, dict] = {q["task_id"]: q for q in self._queries}

        # SQLite in-memory connection (re-created on every reset)
        self._conn: sqlite3.Connection | None = None

        # Episode state
        self._task: dict | None = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._current_score: float = 0.0
        self._schema_name: str = ""
        self._max_steps: int = 5

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def load_task(self, task_name: str) -> None:
        """
        Load a task by name and configure per-episode settings.

        Must be called before ``reset()``.
        """
        self._task = self._query_index[task_name]
        self._schema_name = self._task["schema_name"]
        self._max_steps = {"easy": 3, "medium": 4, "hard": 5}[self._task["difficulty"]]

    def reset(self) -> SQLObservation:
        """
        Start a new episode for the previously loaded task.

        Sets up a fresh in-memory SQLite database, inserts deterministic sample
        rows, and returns an initial SQLObservation with the broken query.
        Call ``load_task()`` first to select the task.
        """
        if self._task is None:
            raise RuntimeError("Call load_task() before reset().")
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._current_score = 0.0

        # Fresh in-memory SQLite instance
        if self._conn is not None:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

        self._setup_schema(self._schema_name)

        return SQLObservation(
            done=False,
            reward=0.0,
            original_query=self._task["broken_query"],
            last_query=self._task["broken_query"],
            execution_result="Episode started. Submit a corrected query.",
            success=False,
            execution_time_ms=0.0,
            result_rows=[],
            feedback=(
                f"Episode started. Task: {self._task['description']} "
                f"Submit your corrected SQL query."
            ),
        )

    def step(self, action: SQLAction) -> SQLObservation:
        """Execute the agent's query and return a full observation."""
        if self._task is None or self._conn is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1

        # ---- Execute the submitted query -----------------------------------
        success = False
        execution_result = ""
        result_rows: list[dict[str, Any]] = []
        exec_time_ms = 0.0

        try:
            cursor = self._conn.cursor()
            t_start = time.perf_counter()
            cursor.execute(action.fixed_query)
            t_end = time.perf_counter()
            exec_time_ms = (t_end - t_start) * 1000.0

            raw_rows = cursor.fetchall()
            result_rows = [dict(row) for row in raw_rows]
            execution_result = f"{len(result_rows)} row(s) returned."
            success = True
        except Exception as exc:
            execution_result = str(exc)

        # ---- Reward --------------------------------------------------------
        reward, feedback = compute_reward(
            task_id=self._task["task_id"],
            action=action,
            execution_result=execution_result,
            success=success,
            result_rows=result_rows,
            execution_time_ms=exec_time_ms,
            step_count=self._step_count,
            max_steps=self._max_steps,
        )
        self._current_score += reward

        # ---- Termination check ---------------------------------------------
        done = (self._current_score >= 1.0) or (self._step_count >= self._max_steps)

        return SQLObservation(
            done=done,
            reward=reward,
            original_query=self._task["broken_query"],
            last_query=action.fixed_query,
            execution_result=execution_result,
            success=success,
            execution_time_ms=exec_time_ms,
            result_rows=result_rows,
            feedback=feedback,
        )

    @property
    def state(self) -> SQLState:
        """Return current episode metadata."""
        if self._task is None:
            raise RuntimeError("Call reset() before accessing state.")
        return SQLState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task["task_id"],
            difficulty=self._task["difficulty"],
            schema=self._schema_name,
            max_steps=self._max_steps,
            current_score=self._current_score,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_schema(self, schema_name: str) -> None:
        """
        Create tables and insert deterministic sample rows for *schema_name*.

        All operations are performed inside a single transaction so the DB is
        always left in a consistent state, making episodes fully reproducible.
        """
        if schema_name not in _DDL:
            raise ValueError(f"Unknown schema: '{schema_name}'. Available: {list(_DDL)}")

        assert self._conn is not None
        with self._conn:
            # Drop any leftover tables from a previous episode on this connection
            # (safety net; should be a fresh :memory: DB at this point)
            for table in _DDL[schema_name]:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")

            # Create tables
            for ddl in _DDL[schema_name].values():
                self._conn.execute(ddl)

            # Insert sample rows
            sample_tables = _SAMPLE_DATA.get(schema_name, {})
            for table, rows in sample_tables.items():
                if not rows:
                    continue
                placeholders = ", ".join(["?"] * len(rows[0]))
                self._conn.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    rows,
                )

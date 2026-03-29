"""
server/app.py — FastAPI application for the SQL Query RL Environment.

Exposes four endpoints:
  POST /reset  — start a new episode
  POST /step   — submit an SQLAction
  GET  /state  — current episode metadata
  GET  /health — liveness check

Run directly:  python server/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make project root importable when running the file directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SQLAction
from server.environment import SQLEnvironment

try:
    from openenv.core.env_server import create_fastapi_app
    _env = SQLEnvironment()
    app: FastAPI = create_fastapi_app(_env)
except Exception:
    # Fallback: plain FastAPI app (useful before openenv is fully installed)
    _env = SQLEnvironment()
    app = FastAPI(
        title="SQL Query RL Environment",
        description="OpenEnv-compatible environment for SQL query repair tasks.",
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# Request body schema for /step
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(task_name: str = "task_1") -> JSONResponse:
    """Start a new episode for the given task."""
    _env.load_task(task_name)
    observation = _env.reset()
    return JSONResponse(content=observation.model_dump())



@app.post("/step")
async def step(action: SQLAction) -> JSONResponse:
    """Submit an SQLAction and receive the resulting observation."""
    observation = _env.step(action)
    return JSONResponse(content=observation.model_dump())


@app.get("/state")
async def state() -> JSONResponse:
    """Return current episode metadata."""
    return JSONResponse(content=_env.state.model_dump())


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness check."""
    return JSONResponse(content={"status": "healthy"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

if __name__ == "__main__":
    main()

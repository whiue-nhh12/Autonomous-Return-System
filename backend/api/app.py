"""FastAPI application factory for an already composed manager graph."""

from fastapi import FastAPI

from agent.orchestration.graph import ManagerAgentGraph
from backend.api.manager import create_manager_router


def create_app(manager: ManagerAgentGraph) -> FastAPI:
    """Create the HTTP app after the host composes configured agent workers."""
    app = FastAPI(title="Automous Return Agent API")
    app.include_router(create_manager_router(manager))
    return app

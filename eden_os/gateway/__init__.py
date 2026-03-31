"""
EDEN OS -- Gateway (Agent 6)
WebSocket/WebRTC Server + REST API Layer.

Exports:
    create_app -- factory function that builds and returns a FastAPI application.
"""

from eden_os.gateway.api_server import create_app

__all__ = ["create_app"]

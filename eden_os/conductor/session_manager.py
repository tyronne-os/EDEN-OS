"""
EDEN OS — Conductor: Session Manager
Manages the lifecycle of conversation sessions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from eden_os.shared.types import AvatarState, SessionConfig


class SessionState(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class Session:
    """Internal representation of a single conversation session."""
    session_id: str
    config: SessionConfig
    state: SessionState = SessionState.CREATED
    avatar_state: AvatarState = AvatarState.IDLE
    engines: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    idle_cache: dict | None = None
    portrait_profile: dict | None = None


class SessionManager:
    """Stores and manages conversation sessions keyed by UUID.

    Usage::

        mgr = SessionManager()
        sid = mgr.create(config)
        session = mgr.get(sid)
        mgr.update_settings(sid, {"expressiveness": 0.9})
        mgr.destroy(sid)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, config: SessionConfig) -> str:
        """Create a new session, returning its UUID session_id.

        If ``config.session_id`` is already set (non-empty), it is honoured;
        otherwise a fresh UUID is generated.
        """
        sid = config.session_id or str(uuid.uuid4())
        config.session_id = sid

        if sid in self._sessions:
            logger.warning("Session '{}' already exists — returning existing", sid)
            return sid

        session = Session(session_id=sid, config=config)
        self._sessions[sid] = session
        logger.info("Session created: {}", sid)
        return sid

    def get(self, session_id: str) -> Session:
        """Retrieve a session by ID. Raises ``KeyError`` if not found."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        return session

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def update_settings(self, session_id: str, settings: dict) -> None:
        """Merge *settings* into the session's config.settings dict."""
        session = self.get(session_id)
        session.config.settings.update(settings)
        logger.info("Session '{}' settings updated: {}", session_id, list(settings.keys()))

    def set_state(self, session_id: str, state: SessionState) -> None:
        session = self.get(session_id)
        old = session.state
        session.state = state
        logger.debug("Session '{}' state: {} -> {}", session_id, old.value, state.value)

    def add_history(self, session_id: str, role: str, content: str) -> None:
        """Append a message to conversation history (sliding window of 40)."""
        session = self.get(session_id)
        session.conversation_history.append({"role": role, "content": content})
        if len(session.conversation_history) > 40:
            session.conversation_history = session.conversation_history[-40:]

    def destroy(self, session_id: str) -> None:
        """Remove a session and release references."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            session.state = SessionState.ENDED
            session.engines.clear()
            logger.info("Session destroyed: {}", session_id)
        else:
            logger.warning("destroy called for unknown session '{}'", session_id)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

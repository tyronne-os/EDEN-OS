"""
EDEN OS -- WebRTC Handler Stub (Agent 6)
Placeholder for future WebRTC peer-connection support.

Currently all real-time streaming goes through the WebSocket fallback.
This module logs stubs so the interface is in place for when aiortc
integration is added.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


class WebRTCHandler:
    """Placeholder WebRTC handler.

    All methods log a notice that WebRTC is not yet active and return
    harmless stub values.  The WebSocket path in ``websocket_handler.py``
    is the active streaming transport for Phase One.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self._peer_connection: Any = None
        self._local_tracks: list[Any] = []
        self._remote_tracks: list[Any] = []
        logger.info(
            f"WebRTCHandler created for session {session_id} — "
            "WebRTC is NOT yet active; using WebSocket fallback."
        )

    # ------------------------------------------------------------------
    # Signaling
    # ------------------------------------------------------------------

    async def create_offer(self) -> dict:
        """Create an SDP offer for a new peer connection.

        Returns a dict with ``sdp`` and ``type`` keys (stub values).
        """
        logger.warning(
            "WebRTCHandler.create_offer() called but WebRTC is not yet "
            "implemented. Returning stub offer."
        )
        return {
            "sdp": "",
            "type": "offer",
            "active": False,
            "message": "WebRTC not yet active. Use WebSocket streaming.",
        }

    async def handle_answer(self, answer: dict) -> bool:
        """Process an SDP answer from the remote peer.

        Parameters
        ----------
        answer:
            Dict with ``sdp`` and ``type`` keys from the remote client.

        Returns True if accepted (always False for now).
        """
        logger.warning(
            "WebRTCHandler.handle_answer() called but WebRTC is not yet "
            "implemented. Answer ignored."
        )
        return False

    async def add_ice_candidate(self, candidate: dict) -> bool:
        """Add a trickle ICE candidate.

        Parameters
        ----------
        candidate:
            ICE candidate dict from the remote peer.
        """
        logger.warning(
            "WebRTCHandler.add_ice_candidate() called but WebRTC is not "
            "yet implemented. Candidate ignored."
        )
        return False

    # ------------------------------------------------------------------
    # Track management
    # ------------------------------------------------------------------

    async def add_track(
        self,
        track: Any,
        kind: str = "video",
    ) -> None:
        """Add a local media track (audio or video) to the peer connection.

        Parameters
        ----------
        track:
            A media track object (e.g. aiortc MediaStreamTrack).
        kind:
            ``"audio"`` or ``"video"``.
        """
        logger.warning(
            f"WebRTCHandler.add_track(kind={kind}) called but WebRTC is "
            "not yet implemented. Track not added."
        )

    async def remove_track(self, track: Any) -> None:
        """Remove a local media track."""
        logger.warning(
            "WebRTCHandler.remove_track() called but WebRTC is not yet "
            "implemented."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the peer connection and release resources."""
        logger.info(
            f"WebRTCHandler.close() for session {self.session_id} — "
            "no active connection to close."
        )
        self._peer_connection = None
        self._local_tracks.clear()
        self._remote_tracks.clear()

    @property
    def is_active(self) -> bool:
        """Whether a WebRTC connection is currently active (always False)."""
        return False

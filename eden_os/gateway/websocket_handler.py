"""
EDEN OS -- WebSocket Streaming Handler (Agent 6)
Bi-directional audio/video streaming over WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from fastapi import WebSocket
from loguru import logger

from eden_os.gateway.audio_capture import AudioCapture
from eden_os.gateway.video_encoder import VideoEncoder
from eden_os.shared.types import AvatarState


class WebSocketHandler:
    """Manages a single bi-directional WebSocket session.

    Receives:
        {"type": "audio",     "data": "<base64_pcm>"}
        {"type": "text",      "content": "hello"}
        {"type": "interrupt"}

    Sends:
        {"type": "video_frame", "data": "<base64>"}
        {"type": "audio",       "data": "<base64_wav>"}
        {"type": "transcript",  "text": "..."}
        {"type": "state",       "value": "speaking"}
    """

    def __init__(
        self,
        session_id: str,
        websocket: WebSocket,
        conductor: Any,
        send_queue_size: int = 256,
        recv_queue_size: int = 256,
    ) -> None:
        self.session_id = session_id
        self.ws = websocket
        self.conductor = conductor

        self._send_queue: asyncio.Queue[dict] = asyncio.Queue(
            maxsize=send_queue_size
        )
        self._recv_queue: asyncio.Queue[dict] = asyncio.Queue(
            maxsize=recv_queue_size
        )
        self._running = False
        self._current_state = AvatarState.IDLE

        self.audio_capture = AudioCapture()
        self.video_encoder = VideoEncoder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until the connection closes."""
        self._running = True
        logger.debug(f"[{self.session_id}] WebSocket handler starting.")

        # Run the receive and send loops concurrently
        recv_task = asyncio.create_task(self._receive_loop())
        send_task = asyncio.create_task(self._send_loop())
        process_task = asyncio.create_task(self._process_loop())

        try:
            done, pending = await asyncio.wait(
                [recv_task, send_task, process_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            # Re-raise any exception from the completed tasks
            for task in done:
                if task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
        finally:
            self._running = False
            logger.debug(f"[{self.session_id}] WebSocket handler stopped.")

    async def close(self) -> None:
        """Signal the handler to stop and close the WebSocket."""
        self._running = False
        try:
            await self.ws.close()
        except Exception:
            pass

    async def handle_interrupt(self) -> None:
        """Handle an interrupt triggered externally (e.g. REST endpoint)."""
        logger.info(f"[{self.session_id}] External interrupt received.")
        await self._set_state(AvatarState.LISTENING)
        try:
            await self.conductor.interrupt(self.session_id)
        except Exception as exc:
            logger.warning(f"[{self.session_id}] Conductor interrupt error: {exc}")

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def send_video_frame(self, frame_b64: str) -> None:
        """Queue a video frame to be sent to the client."""
        await self._enqueue_send({"type": "video_frame", "data": frame_b64})

    async def send_audio(self, audio_b64: str) -> None:
        """Queue an audio chunk to be sent to the client."""
        await self._enqueue_send({"type": "audio", "data": audio_b64})

    async def send_transcript(self, text: str) -> None:
        """Queue a transcript message."""
        await self._enqueue_send({"type": "transcript", "text": text})

    async def send_state(self, state: AvatarState) -> None:
        """Queue a state-change notification."""
        await self._enqueue_send({"type": "state", "value": state.value})

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and push to _recv_queue."""
        while self._running:
            try:
                raw = await self.ws.receive_text()
                msg = json.loads(raw)
                try:
                    self._recv_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning(
                        f"[{self.session_id}] Receive queue full, dropping message."
                    )
            except Exception:
                # WebSocket closed or malformed data
                self._running = False
                break

    async def _send_loop(self) -> None:
        """Drain _send_queue and write to the WebSocket."""
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self._send_queue.get(), timeout=0.1
                )
                await self.ws.send_text(json.dumps(msg))
            except asyncio.TimeoutError:
                continue
            except Exception:
                self._running = False
                break

    async def _process_loop(self) -> None:
        """Drain _recv_queue and dispatch each message to the appropriate handler."""
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self._recv_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            msg_type = msg.get("type")
            try:
                if msg_type == "audio":
                    await self._handle_audio(msg)
                elif msg_type == "text":
                    await self._handle_text(msg)
                elif msg_type == "interrupt":
                    await self._handle_interrupt_msg()
                else:
                    logger.warning(
                        f"[{self.session_id}] Unknown message type: {msg_type}"
                    )
            except Exception as exc:
                logger.error(
                    f"[{self.session_id}] Error processing {msg_type}: {exc}"
                )

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_audio(self, msg: dict) -> None:
        """Process incoming base64 PCM audio from the user."""
        b64_data = msg.get("data", "")
        if not b64_data:
            return

        # Decode and preprocess audio
        chunk = self.audio_capture.process(b64_data)
        if chunk is None:
            return  # Below noise gate

        await self._set_state(AvatarState.LISTENING)

        # Forward to Conductor for ASR + pipeline
        try:
            # If the conductor exposes a handle_audio_chunk method, use it.
            if hasattr(self.conductor, "handle_audio_chunk"):
                result = await self.conductor.handle_audio_chunk(
                    self.session_id, chunk
                )
                if result:
                    await self._dispatch_conductor_result(result)
        except Exception as exc:
            logger.warning(
                f"[{self.session_id}] Conductor audio handling error: {exc}"
            )

    async def _handle_text(self, msg: dict) -> None:
        """Process incoming text from the user (typed input)."""
        content = msg.get("content", "").strip()
        if not content:
            return

        logger.info(f"[{self.session_id}] Text input: {content[:80]}")
        await self._set_state(AvatarState.THINKING)

        try:
            if hasattr(self.conductor, "handle_text_input"):
                result = await self.conductor.handle_text_input(
                    self.session_id, content
                )
                if result:
                    await self._dispatch_conductor_result(result)
        except Exception as exc:
            logger.warning(
                f"[{self.session_id}] Conductor text handling error: {exc}"
            )

    async def _handle_interrupt_msg(self) -> None:
        """Handle an interrupt message from the WebSocket client."""
        logger.info(f"[{self.session_id}] Client interrupt received.")
        await self.handle_interrupt()

    # ------------------------------------------------------------------
    # Conductor result dispatch
    # ------------------------------------------------------------------

    async def _dispatch_conductor_result(self, result: dict) -> None:
        """Dispatch a result dict from the Conductor to outbound messages.

        Expected keys (all optional):
            transcript: str — the avatar's spoken text
            audio: bytes or base64 str — WAV audio to send
            frames: list[np.ndarray] — video frames to encode and send
            state: str — avatar state transition
        """
        if "state" in result:
            new_state = AvatarState(result["state"])
            await self._set_state(new_state)

        if "transcript" in result:
            await self.send_transcript(result["transcript"])

        if "audio" in result:
            audio_data = result["audio"]
            if isinstance(audio_data, (bytes, bytearray)):
                audio_data = base64.b64encode(audio_data).decode("ascii")
            await self.send_audio(audio_data)

        if "frames" in result:
            for frame in result["frames"]:
                encoded = self.video_encoder.encode_frame(frame)
                await self.send_video_frame(encoded)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    async def _set_state(self, new_state: AvatarState) -> None:
        if new_state != self._current_state:
            self._current_state = new_state
            await self.send_state(new_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _enqueue_send(self, msg: dict) -> None:
        try:
            self._send_queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop oldest message to make room
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._send_queue.put_nowait(msg)
            except asyncio.QueueFull:
                logger.warning(
                    f"[{self.session_id}] Send queue overflow, dropping."
                )

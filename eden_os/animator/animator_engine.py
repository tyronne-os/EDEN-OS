"""
EDEN OS — Animator Engine
Main engine composing LivePortrait driver, idle generator, state machine,
audio-to-keypoints bridge, and temporal anchor.
Generates 60fps photorealistic facial animation driven by audio.
"""

import asyncio
import time
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk, AvatarState, VideoFrame
from eden_os.shared.interfaces import IAnimatorEngine
from eden_os.animator.liveportrait_driver import LivePortraitDriver
from eden_os.animator.idle_generator import IdleGenerator
from eden_os.animator.state_machine import AvatarStateMachine
from eden_os.animator.audio_to_keypoints import AudioToKeypoints
from eden_os.animator.eden_temporal_anchor import EdenTemporalAnchor


class AnimatorEngine(IAnimatorEngine):
    """
    Agent 3: Lip-Sync + 4D Motion Engine.
    Generates real-time facial animation driven by audio.
    """

    def __init__(self, models_cache: str = "models_cache/liveportrait", fps: float = 30.0):
        self.driver = LivePortraitDriver(models_cache=models_cache)
        self.idle_gen = IdleGenerator(fps=fps)
        self.state_machine = AvatarStateMachine()
        self.audio_bridge = AudioToKeypoints()
        self.temporal_anchor = EdenTemporalAnchor()

        self._source_image: Optional[np.ndarray] = None
        self._is_initialized = False
        self._current_frame: Optional[VideoFrame] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._conversation_turn = 0
        self.fps = fps

    async def initialize(self, portrait: np.ndarray) -> None:
        """Initialize the animator with a source portrait."""
        await self.driver.load_models()
        self.driver.set_source_image(portrait)
        self._source_image = portrait
        self.temporal_anchor.set_anchor(portrait)
        self._is_initialized = True
        logger.info("Animator engine initialized")

    async def start_idle_loop(self, profile: dict) -> AsyncIterator[VideoFrame]:
        """Generate continuous idle animation frames."""
        if not self._is_initialized:
            raise RuntimeError("Animator not initialized. Call initialize() first.")

        await self.state_machine.transition_to(AvatarState.LISTENING)

        async for frame in self.idle_gen.generate_idle_frames(
            source_image=self._source_image,
            base_keypoints=self.driver.source_keypoints,
            render_fn=self.driver.render_frame,
        ):
            # Apply temporal anchor for identity preservation
            frame.pixels = self.temporal_anchor.stabilize_frame(
                frame.pixels, self._conversation_turn
            )
            self._current_frame = frame
            yield frame

    async def drive_from_audio(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[VideoFrame]:
        """Generate lip-synced animation driven by audio stream."""
        if not self._is_initialized:
            raise RuntimeError("Animator not initialized. Call initialize() first.")

        await self.state_machine.transition_to(AvatarState.SPEAKING)
        frame_interval = 1.0 / self.fps

        async for chunk in audio:
            frame_start = time.monotonic()

            # Extract audio features and convert to keypoint params
            audio_features = self.audio_bridge.process_audio_chunk(chunk.data)

            # Get emotion from chunk if available
            emotion = getattr(chunk, "emotion", None)

            # Apply to LivePortrait keypoints
            keypoints = self.driver.apply_audio_keypoints(audio_features, emotion)

            # Get transition blend if we're mid-transition
            blend = self.state_machine.get_animation_blend()
            if blend.get("is_transitioning"):
                # Blend with idle keypoints during transition
                idle_deltas = self.idle_gen.get_idle_keypoint_deltas(
                    time.monotonic() - frame_start
                )
                idle_kp = self.idle_gen.apply_idle_to_keypoints(
                    self.driver.source_keypoints, idle_deltas
                )
                factor = blend["blend_factor"]
                keypoints = keypoints * factor + idle_kp * (1 - factor)

            # Render frame
            pixels = self.driver.render_frame(keypoints)

            # Apply temporal anchor
            pixels = self.temporal_anchor.stabilize_frame(
                pixels, self._conversation_turn
            )

            frame = VideoFrame(
                pixels=pixels,
                timestamp_ms=time.monotonic() * 1000,
                state=AvatarState.SPEAKING,
                eden_score=1.0,
            )
            self._current_frame = frame
            yield frame

            # If this is the final audio chunk, transition back to listening
            if chunk.is_final:
                self._conversation_turn += 1
                await self.state_machine.transition_to(AvatarState.LISTENING)

            # Maintain frame rate
            elapsed = time.monotonic() - frame_start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

    async def transition_state(
        self, from_state: AvatarState, to_state: AvatarState
    ) -> None:
        """Transition between avatar states."""
        interrupt = (
            from_state == AvatarState.SPEAKING
            and to_state == AvatarState.LISTENING
        )
        await self.state_machine.transition_to(to_state, interrupt=interrupt)

        if interrupt:
            self.audio_bridge.reset()
            self._conversation_turn += 1

    async def get_current_frame(self) -> VideoFrame:
        """Get the most recently rendered frame."""
        if self._current_frame is not None:
            return self._current_frame

        # Return a default frame if nothing has been rendered yet
        if self._source_image is not None:
            return VideoFrame(
                pixels=self._source_image,
                timestamp_ms=time.monotonic() * 1000,
                state=self.state_machine.state,
                eden_score=1.0,
            )

        return VideoFrame(
            pixels=np.zeros((512, 512, 3), dtype=np.uint8),
            timestamp_ms=time.monotonic() * 1000,
            state=AvatarState.IDLE,
            eden_score=0.0,
        )

    def apply_eden_anchor(self, frame: np.ndarray) -> np.ndarray:
        """Apply Eden temporal anchor to a frame."""
        return self.temporal_anchor.stabilize_frame(frame, self._conversation_turn)

    def update_settings(self, settings: dict) -> None:
        """Update animator settings from admin panel sliders."""
        self.driver.update_settings(settings)

        if "eye_contact" in settings:
            self.driver.gaze_lock = float(settings["eye_contact"])

    def get_status(self) -> dict:
        """Get animator status for monitoring."""
        return {
            "initialized": self._is_initialized,
            "state": self.state_machine.state.value,
            "conversation_turn": self._conversation_turn,
            "frame_count": self.driver.frame_count,
            "fps": self.fps,
            "temporal_anchor": self.temporal_anchor.get_status(),
            "state_info": self.state_machine.get_state_info(),
        }

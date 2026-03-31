"""
EDEN OS — Avatar State Machine
Manages transitions between LISTENING, THINKING, and SPEAKING states.
Implements the KV-Recache Interruption Protocol for seamless transitions.
"""

import asyncio
import time
from typing import Callable, Optional

from loguru import logger

from eden_os.shared.types import AvatarState


class AvatarStateMachine:
    """
    Manages the three states of avatar presence:
    LISTENING → THINKING → SPEAKING → LISTENING (loop)
    Plus interrupt handling for SPEAKING → LISTENING.
    """

    def __init__(self):
        self._state = AvatarState.IDLE
        self._previous_state = AvatarState.IDLE
        self._state_enter_time: float = time.monotonic()
        self._transition_progress: float = 1.0  # 0.0 = just started, 1.0 = complete
        self._transition_duration: float = 0.2  # seconds for smooth transitions
        self._is_interrupted: bool = False

        # Callbacks for state transitions
        self._on_enter_callbacks: dict[AvatarState, list[Callable]] = {
            s: [] for s in AvatarState
        }
        self._on_exit_callbacks: dict[AvatarState, list[Callable]] = {
            s: [] for s in AvatarState
        }

        # Transition parameters for animation blending
        self._transition_params: dict = {}

    @property
    def state(self) -> AvatarState:
        return self._state

    @property
    def previous_state(self) -> AvatarState:
        return self._previous_state

    @property
    def time_in_state(self) -> float:
        """Seconds since entering current state."""
        return time.monotonic() - self._state_enter_time

    @property
    def transition_progress(self) -> float:
        """Progress of current transition (0.0 to 1.0)."""
        if self._transition_progress >= 1.0:
            return 1.0
        elapsed = time.monotonic() - self._state_enter_time
        progress = min(1.0, elapsed / self._transition_duration)
        return progress

    @property
    def is_transitioning(self) -> bool:
        return self.transition_progress < 1.0

    def on_enter(self, state: AvatarState, callback: Callable) -> None:
        """Register callback for entering a state."""
        self._on_enter_callbacks[state].append(callback)

    def on_exit(self, state: AvatarState, callback: Callable) -> None:
        """Register callback for exiting a state."""
        self._on_exit_callbacks[state].append(callback)

    async def transition_to(self, new_state: AvatarState, interrupt: bool = False) -> None:
        """
        Transition to a new state with smooth blending.

        Args:
            new_state: Target state
            interrupt: If True, this is an interrupt transition (faster, uses KV-recache)
        """
        if new_state == self._state and not interrupt:
            return

        old_state = self._state
        self._previous_state = old_state
        self._is_interrupted = interrupt

        # Set transition duration based on type
        if interrupt:
            # KV-Recache: ultra-fast transition for interrupts
            self._transition_duration = 0.1  # 100ms per spec
            logger.info(f"INTERRUPT: {old_state.value} → {new_state.value} (KV-recache)")
        else:
            self._transition_duration = self._get_transition_duration(old_state, new_state)
            logger.info(f"State: {old_state.value} → {new_state.value}")

        # Set transition-specific animation parameters
        self._transition_params = self._get_transition_params(old_state, new_state, interrupt)

        # Fire exit callbacks
        for cb in self._on_exit_callbacks.get(old_state, []):
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Exit callback error: {e}")

        # Update state
        self._state = new_state
        self._state_enter_time = time.monotonic()
        self._transition_progress = 0.0

        # Fire enter callbacks
        for cb in self._on_enter_callbacks.get(new_state, []):
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Enter callback error: {e}")

    def _get_transition_duration(self, from_state: AvatarState, to_state: AvatarState) -> float:
        """Get transition duration based on state pair."""
        transitions = {
            (AvatarState.LISTENING, AvatarState.THINKING): 0.3,
            (AvatarState.THINKING, AvatarState.SPEAKING): 0.2,
            (AvatarState.SPEAKING, AvatarState.LISTENING): 0.3,
            (AvatarState.IDLE, AvatarState.LISTENING): 0.5,
        }
        return transitions.get((from_state, to_state), 0.3)

    def _get_transition_params(
        self, from_state: AvatarState, to_state: AvatarState, interrupt: bool
    ) -> dict:
        """Get animation blending parameters for the transition."""
        params = {
            "blend_factor": 0.0,
            "brow_raise": 0.0,
            "inhale": False,
            "mouth_close": False,
        }

        if from_state == AvatarState.LISTENING and to_state == AvatarState.THINKING:
            # Subtle inhale, slight brow raise
            params["inhale"] = True
            params["brow_raise"] = 0.3

        elif from_state == AvatarState.THINKING and to_state == AvatarState.SPEAKING:
            # Open mouth, begin lip-sync
            params["mouth_close"] = False

        elif to_state == AvatarState.LISTENING:
            # Close mouth, return to idle
            params["mouth_close"] = True

            if interrupt:
                # KV-Recache: preserve current face position anchors
                params["preserve_anchors"] = True

        return params

    def get_animation_blend(self) -> dict:
        """
        Get current animation blend parameters based on transition progress.
        Used by the animator to smoothly blend between states.
        """
        progress = self.transition_progress
        params = self._transition_params.copy()
        params["blend_factor"] = progress
        params["state"] = self._state
        params["previous_state"] = self._previous_state
        params["is_transitioning"] = self.is_transitioning
        return params

    def get_state_info(self) -> dict:
        """Get current state information for API/frontend."""
        return {
            "state": self._state.value,
            "previous_state": self._previous_state.value,
            "time_in_state": self.time_in_state,
            "is_transitioning": self.is_transitioning,
            "transition_progress": self.transition_progress,
            "is_interrupted": self._is_interrupted,
        }

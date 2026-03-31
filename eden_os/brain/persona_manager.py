"""
EDEN OS -- Brain Engine: Persona Manager
Loads YAML persona templates. Returns system prompt, emotional baseline,
voice config, and appearance config. Validates template schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


# Required keys at each level of the template schema
_REQUIRED_AGENT_KEYS = {"name", "role", "persona", "system_prompt"}
_REQUIRED_PERSONA_KEYS = {"tone", "emotional_baseline"}
_DEFAULT_EMOTIONAL_BASELINE = {
    "joy": 0.5,
    "sadness": 0.0,
    "confidence": 0.7,
    "urgency": 0.0,
    "warmth": 0.6,
}


class PersonaValidationError(Exception):
    """Raised when a persona template fails schema validation."""


class PersonaManager:
    """Loads and serves persona configuration from a validated YAML template."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._system_prompt: str = ""
        self._emotional_baseline: dict[str, float] = dict(_DEFAULT_EMOTIONAL_BASELINE)
        self._voice_config: dict[str, Any] = {}
        self._appearance_config: dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, template_config: dict[str, Any]) -> None:
        """
        Load and validate a persona template config dict.

        Args:
            template_config: Parsed YAML dict with an ``agent`` top-level key.

        Raises:
            PersonaValidationError: If the template fails schema validation.
        """
        self._validate(template_config)
        agent = template_config["agent"]

        self._config = template_config
        self._system_prompt = agent["system_prompt"].strip()

        persona = agent["persona"]
        self._emotional_baseline = self._normalise_emotions(
            persona.get("emotional_baseline", _DEFAULT_EMOTIONAL_BASELINE)
        )

        self._voice_config = agent.get("voice", {})
        self._appearance_config = agent.get("appearance", {})
        self._loaded = True

        logger.info(
            "Persona loaded: name='{}' role='{}'",
            agent["name"],
            agent["role"],
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def emotional_baseline(self) -> dict[str, float]:
        return dict(self._emotional_baseline)

    @property
    def voice_config(self) -> dict[str, Any]:
        return dict(self._voice_config)

    @property
    def appearance_config(self) -> dict[str, Any]:
        return dict(self._appearance_config)

    @property
    def name(self) -> str:
        if not self._loaded:
            return "EVE"
        return self._config["agent"]["name"]

    @property
    def role(self) -> str:
        if not self._loaded:
            return "conversational assistant"
        return self._config["agent"]["role"]

    def get_full_config(self) -> dict[str, Any]:
        """Return the full validated config dict."""
        return dict(self._config)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(config: dict[str, Any]) -> None:
        """Validate the template against the expected schema."""
        if not isinstance(config, dict):
            raise PersonaValidationError("Template must be a dict")

        if "agent" not in config:
            raise PersonaValidationError("Template missing top-level 'agent' key")

        agent = config["agent"]
        if not isinstance(agent, dict):
            raise PersonaValidationError("'agent' must be a dict")

        missing = _REQUIRED_AGENT_KEYS - set(agent.keys())
        if missing:
            raise PersonaValidationError(
                f"Agent config missing required keys: {missing}"
            )

        persona = agent["persona"]
        if not isinstance(persona, dict):
            raise PersonaValidationError("'persona' must be a dict")

        missing_persona = _REQUIRED_PERSONA_KEYS - set(persona.keys())
        if missing_persona:
            raise PersonaValidationError(
                f"Persona config missing required keys: {missing_persona}"
            )

        # Validate emotional_baseline values are 0.0-1.0
        baseline = persona.get("emotional_baseline", {})
        if isinstance(baseline, dict):
            for key, val in baseline.items():
                if not isinstance(val, (int, float)):
                    raise PersonaValidationError(
                        f"emotional_baseline['{key}'] must be a number, got {type(val).__name__}"
                    )
                if not 0.0 <= float(val) <= 1.0:
                    raise PersonaValidationError(
                        f"emotional_baseline['{key}'] must be 0.0-1.0, got {val}"
                    )

    @staticmethod
    def _normalise_emotions(raw: dict[str, Any]) -> dict[str, float]:
        """Ensure all five canonical emotion keys exist with float values."""
        out = dict(_DEFAULT_EMOTIONAL_BASELINE)
        for k, v in raw.items():
            out[k] = float(v)
        return out

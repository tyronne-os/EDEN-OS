"""
EDEN OS -- Brain Engine: Template Loader
Loads and validates YAML templates from the templates/ directory.
Lists available templates. Returns validated config dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from eden_os.brain.persona_manager import PersonaManager, PersonaValidationError


# Default templates directory (project root / templates)
_DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


class TemplateLoader:
    """
    Discovers, loads, and validates YAML persona templates.

    Templates live in the ``templates/`` directory at the project root.
    Each ``.yaml`` file must conform to the agent template schema
    validated by :class:`PersonaManager`.
    """

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        self.templates_dir = Path(templates_dir) if templates_dir else _DEFAULT_TEMPLATES_DIR
        logger.info("TemplateLoader using directory: {}", self.templates_dir)

    def list_templates(self) -> list[str]:
        """
        List available template names (without extension).

        Returns:
            Sorted list of template names, e.g. ``["default", "medical_office"]``.
        """
        if not self.templates_dir.is_dir():
            logger.warning("Templates directory not found: {}", self.templates_dir)
            return []

        templates = sorted(
            p.stem for p in self.templates_dir.glob("*.yaml")
        )
        logger.debug("Available templates: {}", templates)
        return templates

    def load(self, template_name: str) -> dict[str, Any]:
        """
        Load and validate a template by name.

        Args:
            template_name: Template name (without ``.yaml`` extension),
                or a full file path.

        Returns:
            Validated config dict with ``agent`` top-level key.

        Raises:
            FileNotFoundError: If template file does not exist.
            PersonaValidationError: If template fails schema validation.
            yaml.YAMLError: If YAML parsing fails.
        """
        path = self._resolve_path(template_name)

        logger.info("Loading template: {}", path)
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        if config is None:
            raise PersonaValidationError(f"Template file is empty: {path}")

        # Validate via PersonaManager's static validator
        PersonaManager._validate(config)
        logger.info("Template '{}' loaded and validated", template_name)
        return config

    def load_from_path(self, path: str | Path) -> dict[str, Any]:
        """
        Load and validate a template from an explicit file path.

        Args:
            path: Absolute or relative path to a YAML template file.

        Returns:
            Validated config dict.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        if config is None:
            raise PersonaValidationError(f"Template file is empty: {path}")

        PersonaManager._validate(config)
        logger.info("Template loaded from path: {}", path)
        return config

    def _resolve_path(self, template_name: str) -> Path:
        """Resolve a template name to a file path."""
        # If it already looks like a path, use it directly
        candidate = Path(template_name)
        if candidate.suffix == ".yaml" and candidate.exists():
            return candidate

        # Look in templates directory
        path = self.templates_dir / f"{template_name}.yaml"
        if path.exists():
            return path

        # Try without adding extension (user passed "foo.yaml")
        path_raw = self.templates_dir / template_name
        if path_raw.exists():
            return path_raw

        raise FileNotFoundError(
            f"Template '{template_name}' not found in {self.templates_dir}. "
            f"Available: {self.list_templates()}"
        )

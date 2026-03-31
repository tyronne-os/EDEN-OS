"""
EDEN OS — Genesis Engine (Agent 1: Portrait-to-4D)
Composes all sub-modules into a single GenesisEngine that implements
the IGenesisEngine interface.

Usage:
    from eden_os.genesis import GenesisEngine
    engine = GenesisEngine()
    result = await engine.process_upload(image)
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from eden_os.shared.interfaces import IGenesisEngine
from eden_os.shared.types import EdenValidationResult

from eden_os.genesis.portrait_engine import PortraitEngine
from eden_os.genesis.eden_protocol_validator import EdenProtocolValidator
from eden_os.genesis.latent_encoder import LatentEncoder
from eden_os.genesis.preload_cache import PreloadCache
from eden_os.genesis.skin_realism_agent import SkinRealismAgent


class GenesisEngine(IGenesisEngine):
    """Agent 1 — Portrait-to-4D Engine.

    Composes:
        PortraitEngine          – face detection, alignment, lighting normalisation
        EdenProtocolValidator   – 0.3 deviation skin texture fidelity check
        LatentEncoder           – portrait → animation-ready latent vector
        PreloadCache            – pre-computed idle animation seed data
    """

    def __init__(self, latent_dim: int = 512, num_idle_seeds: int = 8) -> None:
        self._portrait = PortraitEngine()
        self._validator = EdenProtocolValidator()
        self._encoder = LatentEncoder(latent_dim=latent_dim)
        self._cache = PreloadCache(num_seeds=num_idle_seeds)
        self.skin_agent = SkinRealismAgent()
        logger.info("GenesisEngine initialised (latent_dim={}, idle_seeds={})",
                    latent_dim, num_idle_seeds)

    # ------------------------------------------------------------------
    # IGenesisEngine interface
    # ------------------------------------------------------------------

    async def process_upload(self, image: np.ndarray) -> dict:
        """Process uploaded portrait: face detection, alignment, enhancement.

        Parameters
        ----------
        image : np.ndarray – RGB uint8 image (H, W, 3).

        Returns
        -------
        dict with keys: aligned_face, landmarks, bbox, original_image.
        """
        return await self._portrait.process(image)

    async def validate_eden_protocol(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.3,
    ) -> EdenValidationResult:
        """Validate skin texture fidelity against reference.

        Parameters
        ----------
        generated : np.ndarray – RGB uint8 generated portrait.
        reference : np.ndarray – RGB uint8 reference portrait.
        threshold : float      – max allowed deviation (default 0.3).

        Returns
        -------
        EdenValidationResult with .passed, .score, .feedback.
        """
        return await self._validator.validate(generated, reference, threshold)

    async def encode_to_latent(self, portrait: np.ndarray) -> np.ndarray:
        """Encode portrait to animation-engine-compatible latent vector.

        Parameters
        ----------
        portrait : np.ndarray – RGB uint8 image.

        Returns
        -------
        np.ndarray of shape (latent_dim,) float32.
        """
        return await self._encoder.encode(portrait)

    async def precompute_idle_cache(self, profile: dict) -> dict:
        """Pre-compute idle animation seed frames and breathing cycle.

        Parameters
        ----------
        profile : dict – must contain 'aligned_face' (np.ndarray 512x512x3).

        Returns
        -------
        dict with 'seed_frames' and 'breathing_cycle' lists.
        """
        return await self._cache.compute(profile)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release resources held by sub-engines."""
        self._portrait.close()
        logger.info("GenesisEngine closed")


__all__ = ["GenesisEngine", "SkinRealismAgent"]

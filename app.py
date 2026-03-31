"""
EDEN OS — Entry Point
Boots the full operating system and serves:
- EDEN Studio frontend at /
- EDEN OS API at /api/v1/
- WebSocket stream at /api/v1/sessions/{id}/stream

Deploy to HuggingFace Spaces as Docker SDK.
Space ID: AIBRUH/eden-os
"""

import os
import sys
import time
from pathlib import Path

import yaml
import uvicorn
from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
    level=os.getenv("EDEN_LOG_LEVEL", "INFO"),
)


def detect_hardware() -> dict:
    """Auto-detect GPU hardware and select profile."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = vram_bytes / (1024 ** 3)

            if "H100" in gpu_name or "A100" in gpu_name:
                profile = "h100_cinematic"
            elif "4090" in gpu_name:
                profile = "rtx4090_production"
            elif "3090" in gpu_name or "3080" in gpu_name:
                profile = "rtx3090_standard"
            elif "L4" in gpu_name or "T4" in gpu_name:
                profile = "l4_cloud"
            else:
                profile = "l4_cloud"  # default GPU profile

            return {
                "gpu": gpu_name,
                "vram_gb": round(vram_gb, 1),
                "gpu_available": True,
                "profile": profile,
            }
    except Exception:
        pass

    return {
        "gpu": "none",
        "vram_gb": 0,
        "gpu_available": False,
        "profile": "cpu_edge",
    }


def load_config(hardware_profile: str = "auto", models_cache: str = "models_cache") -> dict:
    """Load and merge configuration."""
    config_dir = Path(__file__).parent / "config"

    # Load default config
    default_path = config_dir / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load hardware profile
    if hardware_profile != "auto":
        profile_path = config_dir / "hardware_profiles" / f"{hardware_profile}.yaml"
        if profile_path.exists():
            with open(profile_path) as f:
                profile = yaml.safe_load(f)
            # Merge profile into config (profile overrides defaults)
            _deep_merge(config, profile)

    config["models_cache"] = models_cache
    config["hardware_profile"] = hardware_profile
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def boot():
    """Boot EDEN OS."""
    start_time = time.monotonic()

    logger.info("=" * 55)
    logger.info(" EDEN OS v1.0 — BOOTING")
    logger.info("=" * 55)

    # Step 1: Detect hardware
    hw = detect_hardware()
    hardware_profile = os.getenv("EDEN_HARDWARE_PROFILE", "auto")
    if hardware_profile == "auto":
        hardware_profile = hw["profile"]
    logger.info(f"Hardware: {hw['gpu']} — Profile: {hardware_profile}")

    # Step 2: Load configuration
    models_cache = os.getenv("EDEN_MODELS_CACHE", "models_cache")
    config = load_config(hardware_profile, models_cache)

    # Step 3: Create FastAPI app
    from eden_os.gateway import create_app
    host = os.getenv("EDEN_HOST", "0.0.0.0")
    port = int(os.getenv("EDEN_PORT", "7860"))

    app = create_app(
        host=host,
        port=port,
        hardware_profile=hardware_profile,
        models_cache=models_cache,
    )

    # Store config and hardware info on app state
    app.state.config = config
    app.state.hardware = hw

    boot_time = time.monotonic() - start_time

    logger.info("=" * 55)
    logger.info(f" EDEN OS v1.0 — LIVE")
    logger.info(f" URL: http://{host}:{port}")
    logger.info(f" API: http://{host}:{port}/api/v1/docs")
    logger.info(f" Hardware: {hw['gpu']} — Profile: {hardware_profile}")
    logger.info(f" Eden Protocol: ACTIVE — Threshold: 0.3")
    logger.info(f" Boot time: {boot_time:.1f}s")
    logger.info(f" OWN THE SCIENCE.")
    logger.info("=" * 55)

    return app, host, port


# Create app at module level for uvicorn import
app, _host, _port = boot()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=_host,
        port=_port,
        ws_max_size=16 * 1024 * 1024,
        log_level="info",
    )

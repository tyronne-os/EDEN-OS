"""
EDEN OS — Model Setup Script
Downloads required model weights from HuggingFace Hub.
"""

import argparse
import os
from pathlib import Path

from loguru import logger


# Models to download (repo_id, local_subdir, description)
MODELS = [
    {
        "repo_id": "KwaiVGI/LivePortrait",
        "subdir": "liveportrait",
        "description": "LivePortrait — real-time facial animation",
        "optional": True,
    },
    {
        "repo_id": "openai/whisper-large-v3-turbo",
        "subdir": "whisper-large-v3-turbo",
        "description": "Whisper Large v3 Turbo — ASR",
        "optional": True,
    },
    {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "subdir": "all-MiniLM-L6-v2",
        "description": "MiniLM — sentence embeddings for RAG",
        "optional": False,
    },
]


def setup_models(cache_dir: str = "models_cache", essential_only: bool = False):
    """Download model weights to cache directory."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model cache directory: {cache_path.absolute()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        return

    for model in MODELS:
        if essential_only and model.get("optional"):
            logger.info(f"Skipping optional model: {model['description']}")
            continue

        target = cache_path / model["subdir"]
        if target.exists() and any(target.iterdir()):
            logger.info(f"Already cached: {model['description']}")
            continue

        logger.info(f"Downloading: {model['description']} ({model['repo_id']})")
        try:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=str(target),
                token=os.getenv("HF_TOKEN"),
            )
            logger.info(f"Downloaded: {model['description']}")
        except Exception as e:
            if model.get("optional"):
                logger.warning(f"Optional model failed (will use fallback): {e}")
            else:
                logger.error(f"Required model failed: {e}")

    logger.info("Model setup complete")


def validate_gpu():
    """Check GPU availability and capabilities."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return {"gpu": gpu_name, "vram_gb": vram_gb, "available": True}
        else:
            logger.warning("No GPU detected — running in CPU mode")
            return {"gpu": "none", "vram_gb": 0, "available": False}
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return {"gpu": "none", "vram_gb": 0, "available": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDEN OS — Model Setup")
    parser.add_argument("--cache-dir", default="models_cache", help="Model cache directory")
    parser.add_argument("--essential-only", action="store_true", help="Only download essential models")
    parser.add_argument("--validate-gpu", action="store_true", help="Check GPU and exit")
    args = parser.parse_args()

    if args.validate_gpu:
        validate_gpu()
    else:
        setup_models(args.cache_dir, args.essential_only)

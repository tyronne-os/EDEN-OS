"""
EDEN OS — Tiered VRAM & Storage Strategy
3-tier model management: GPU VRAM → Local Storage → HuggingFace Cloud

Tier 0 (HOT)  : GPU VRAM — active inference models (LivePortrait, Whisper, TTS)
Tier 1 (WARM) : Seagate 5TB / Local SSD — pre-downloaded weights, instant load
Tier 2 (COLD) : HuggingFace Hub (AIBRUH, 1TB) — persistent cloud cache, pull on demand

Strategy:
- Only keep the ACTIVE pipeline models in VRAM at any time
- Swap models in/out of VRAM based on conversation state
- Pre-fetch next-likely models from Seagate to RAM during idle
- Use HF Hub as infinite cold storage — never re-download what's cached

On a 24GB GPU:
  LivePortrait (4GB) + Whisper (3GB) + Kokoro TTS (1GB) + Brain overhead (2GB) = 10GB
  Leaves 14GB headroom for FLUX, HunyuanVideo, or larger TTS models

On CPU-only (this machine):
  All models run on CPU/RAM — Seagate becomes the primary model cache
  HF Hub serves as backup if Seagate is disconnected
"""

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


class StorageTier(Enum):
    VRAM = "vram"       # GPU memory — active inference
    RAM = "ram"         # System RAM — loaded but not on GPU
    LOCAL = "local"     # Seagate 5TB or local SSD
    CLOUD = "cloud"     # HuggingFace Hub (AIBRUH)


class ModelPriority(Enum):
    CRITICAL = 0    # Must always be in VRAM (LivePortrait, active TTS)
    HIGH = 1        # Load to VRAM on demand, keep in RAM (Whisper, emotion router)
    MEDIUM = 2      # Keep on local disk, load when needed (FLUX, StyleTTS2)
    LOW = 3         # Keep on HF Hub, download on first use (HunyuanVideo, Qwen)


@dataclass
class ModelSlot:
    """Tracks a model's location across the storage hierarchy."""
    name: str
    repo_id: str
    size_gb: float
    priority: ModelPriority
    current_tier: StorageTier = StorageTier.CLOUD
    local_path: Optional[str] = None
    vram_loaded: bool = False
    last_used: float = 0.0
    load_time_ms: float = 0.0
    engine: str = ""  # which engine owns this model


# ═══════════════════════════════════════════════════════════════
# EDEN OS Model Registry — every model the system can use
# ═══════════════════════════════════════════════════════════════
MODEL_REGISTRY: list[dict] = [
    # ── CRITICAL: Always in VRAM during conversation ──
    {
        "name": "liveportrait",
        "repo_id": "KwaiVGI/LivePortrait",
        "size_gb": 4.0,
        "priority": ModelPriority.CRITICAL,
        "engine": "animator",
    },
    {
        "name": "kokoro-tts",
        "repo_id": "hexgrad/Kokoro-82M",
        "size_gb": 0.5,
        "priority": ModelPriority.CRITICAL,
        "engine": "voice",
    },
    {
        "name": "silero-vad",
        "repo_id": "snakers5/silero-vad",
        "size_gb": 0.1,
        "priority": ModelPriority.CRITICAL,
        "engine": "voice",
    },

    # ── HIGH: Load on demand, keep in RAM ──
    {
        "name": "whisper-large-v3-turbo",
        "repo_id": "openai/whisper-large-v3-turbo",
        "size_gb": 3.0,
        "priority": ModelPriority.HIGH,
        "engine": "voice",
    },
    {
        "name": "insightface",
        "repo_id": "deepinsight/insightface",
        "size_gb": 0.5,
        "priority": ModelPriority.HIGH,
        "engine": "genesis",
    },
    {
        "name": "minilm-embeddings",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "size_gb": 0.1,
        "priority": ModelPriority.HIGH,
        "engine": "scholar",
    },

    # ── MEDIUM: Keep on Seagate, load when needed ──
    {
        "name": "cosyvoice2",
        "repo_id": "FunAudioLLM/CosyVoice2-0.5B",
        "size_gb": 2.0,
        "priority": ModelPriority.MEDIUM,
        "engine": "voice",
    },
    {
        "name": "styletts2",
        "repo_id": "yl4579/StyleTTS2",
        "size_gb": 4.0,
        "priority": ModelPriority.MEDIUM,
        "engine": "voice",
    },
    {
        "name": "flux-schnell",
        "repo_id": "black-forest-labs/FLUX.1-schnell",
        "size_gb": 14.0,
        "priority": ModelPriority.MEDIUM,
        "engine": "genesis",
    },
    {
        "name": "ip-adapter-faceid",
        "repo_id": "h94/IP-Adapter-FaceID",
        "size_gb": 2.0,
        "priority": ModelPriority.MEDIUM,
        "engine": "genesis",
    },
    {
        "name": "realesrgan",
        "repo_id": "ai-forever/Real-ESRGAN",
        "size_gb": 0.5,
        "priority": ModelPriority.MEDIUM,
        "engine": "genesis",
    },

    # ── LOW: Keep on HF Hub, pull when first needed ──
    {
        "name": "hunyuan-avatar",
        "repo_id": "tencent/HunyuanVideo-Avatar",
        "size_gb": 16.0,
        "priority": ModelPriority.LOW,
        "engine": "animator",
    },
    {
        "name": "musetalk",
        "repo_id": "TMElyralab/MuseTalk",
        "size_gb": 4.0,
        "priority": ModelPriority.LOW,
        "engine": "animator",
    },
    {
        "name": "qwen3-8b-gguf",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "size_gb": 5.0,
        "priority": ModelPriority.LOW,
        "engine": "brain",
    },
    {
        "name": "qwen2.5-vl-7b",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb": 15.0,
        "priority": ModelPriority.LOW,
        "engine": "scholar",
    },
    {
        "name": "flux-pro",
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "size_gb": 22.0,
        "priority": ModelPriority.LOW,
        "engine": "genesis",
    },
]


class VRAMStrategy:
    """
    Manages model placement across the 3-tier storage hierarchy.

    Tier 0 — GPU VRAM: Active models only. Strict budget enforcement.
    Tier 1 — Seagate 5TB / Local SSD: Fast local cache. Models load in seconds.
    Tier 2 — HuggingFace Hub (AIBRUH): 1TB cloud. Models download in minutes.

    The strategy ensures:
    1. CRITICAL models are always loaded for real-time conversation
    2. Models swap in/out of VRAM automatically based on pipeline state
    3. Seagate acts as a local CDN — pre-fetched models skip HF downloads
    4. When Seagate is disconnected, falls back to local SSD cache
    """

    # Storage paths
    SEAGATE_MOUNT = "/mnt/seagate5tb"
    SEAGATE_CACHE = "/mnt/seagate5tb/eden-os/models"
    LOCAL_CACHE = os.path.expanduser("~/EDEN-OS/models_cache")
    HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    HF_NAMESPACE = "AIBRUH"

    def __init__(self, vram_budget_gb: float = 0.0, ram_budget_gb: float = 3.0):
        """
        Args:
            vram_budget_gb: Available GPU VRAM in GB (0 = CPU only)
            ram_budget_gb: RAM budget for model loading (conservative default)
        """
        self.vram_budget_gb = vram_budget_gb
        self.ram_budget_gb = ram_budget_gb
        self.vram_used_gb = 0.0
        self.ram_used_gb = 0.0

        # Build model slots from registry
        self.models: dict[str, ModelSlot] = {}
        for m in MODEL_REGISTRY:
            slot = ModelSlot(
                name=m["name"],
                repo_id=m["repo_id"],
                size_gb=m["size_gb"],
                priority=m["priority"],
                engine=m["engine"],
            )
            self.models[m["name"]] = slot

        # Detect storage tiers
        self.seagate_available = self._detect_seagate()
        self.gpu_available = vram_budget_gb > 0

        # Scan existing caches
        self._scan_local_cache()

        logger.info(
            f"VRAM Strategy initialized: "
            f"VRAM={vram_budget_gb:.1f}GB, RAM={ram_budget_gb:.1f}GB, "
            f"Seagate={'CONNECTED' if self.seagate_available else 'DISCONNECTED'}, "
            f"GPU={'YES' if self.gpu_available else 'CPU-only'}"
        )

    # ── Storage Detection ──────────────────────────────────────────

    def _detect_seagate(self) -> bool:
        """Detect if Seagate 5TB is mounted and accessible."""
        seagate_path = Path(self.SEAGATE_MOUNT)
        if seagate_path.exists() and seagate_path.is_dir():
            try:
                # Check if it's actually mounted (not just an empty dir)
                stat = os.statvfs(str(seagate_path))
                total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

                if total_gb > 100:  # Seagate is 5TB, sanity check
                    logger.info(
                        f"Seagate 5TB detected: {total_gb:.0f}GB total, "
                        f"{free_gb:.0f}GB free"
                    )
                    # Ensure cache directory exists
                    Path(self.SEAGATE_CACHE).mkdir(parents=True, exist_ok=True)
                    return True
            except OSError:
                pass

        # Try Windows PowerShell detection via WSL
        try:
            import subprocess
            result = subprocess.run(
                ["powershell.exe", "-Command",
                 "(Get-Volume -FileSystemLabel 'SEAGATE5TB').DriveLetter"],
                capture_output=True, text=True, timeout=5,
            )
            letter = result.stdout.strip()
            if letter:
                wsl_path = f"/mnt/{letter.lower()}"
                if Path(wsl_path).exists():
                    self.SEAGATE_MOUNT = wsl_path
                    self.SEAGATE_CACHE = f"{wsl_path}/eden-os/models"
                    Path(self.SEAGATE_CACHE).mkdir(parents=True, exist_ok=True)
                    logger.info(f"Seagate found via PowerShell at {wsl_path}")
                    return True
        except Exception:
            pass

        logger.info("Seagate 5TB not detected — using local SSD cache only")
        return False

    def _scan_local_cache(self) -> None:
        """Scan local cache directories for already-downloaded models."""
        for name, slot in self.models.items():
            # Check local cache
            local_path = Path(self.LOCAL_CACHE) / name
            if local_path.exists() and any(local_path.iterdir()):
                slot.current_tier = StorageTier.LOCAL
                slot.local_path = str(local_path)
                continue

            # Check Seagate cache
            if self.seagate_available:
                seagate_path = Path(self.SEAGATE_CACHE) / name
                if seagate_path.exists() and any(seagate_path.iterdir()):
                    slot.current_tier = StorageTier.LOCAL
                    slot.local_path = str(seagate_path)
                    continue

            # Check HF cache
            hf_path = Path(self.HF_CACHE) / f"models--{slot.repo_id.replace('/', '--')}"
            if hf_path.exists():
                slot.current_tier = StorageTier.LOCAL
                slot.local_path = str(hf_path)
                continue

            # Still on cloud
            slot.current_tier = StorageTier.CLOUD

    # ── VRAM Management ────────────────────────────────────────────

    def get_vram_plan(self, pipeline_type: str = "conversation") -> dict:
        """
        Generate an optimal VRAM allocation plan for the given pipeline.

        Returns dict with:
          - load_to_vram: models to load into GPU
          - keep_in_ram: models to keep in system RAM
          - prefetch_to_local: models to download from HF to Seagate/SSD
          - estimated_vram_usage: total GB
        """
        plans = {
            "conversation": self._plan_conversation(),
            "portrait_generation": self._plan_portrait(),
            "knowledge_ingestion": self._plan_knowledge(),
            "cinematic": self._plan_cinematic(),
        }
        return plans.get(pipeline_type, self._plan_conversation())

    def _plan_conversation(self) -> dict:
        """
        Real-time conversation pipeline.
        Must fit: LivePortrait + TTS + VAD + Whisper (on demand)
        """
        vram_models = []
        ram_models = []
        remaining = self.vram_budget_gb

        # Priority order for conversation
        conversation_models = [
            "liveportrait",       # 4.0GB — CRITICAL, always on
            "kokoro-tts",         # 0.5GB — CRITICAL, always on
            "silero-vad",         # 0.1GB — CRITICAL, always on
            "whisper-large-v3-turbo",  # 3.0GB — HIGH, load when listening
            "minilm-embeddings",  # 0.1GB — HIGH, for RAG retrieval
            "insightface",        # 0.5GB — needed for initial setup only
        ]

        for name in conversation_models:
            slot = self.models.get(name)
            if not slot:
                continue

            if self.gpu_available and remaining >= slot.size_gb:
                vram_models.append(name)
                remaining -= slot.size_gb
            else:
                ram_models.append(name)

        return {
            "pipeline": "conversation",
            "load_to_vram": vram_models,
            "keep_in_ram": ram_models,
            "prefetch_to_local": [
                n for n in conversation_models
                if self.models.get(n) and self.models[n].current_tier == StorageTier.CLOUD
            ],
            "estimated_vram_gb": self.vram_budget_gb - remaining,
            "vram_headroom_gb": remaining,
            "swap_strategy": (
                "LivePortrait stays resident. Whisper loads on LISTENING state, "
                "unloads during SPEAKING if VRAM is tight. TTS + VAD never leave VRAM."
            ),
        }

    def _plan_portrait(self) -> dict:
        """Portrait generation (Genesis). Temporarily unload conversation models."""
        vram_models = []
        remaining = self.vram_budget_gb

        portrait_models = [
            "flux-schnell",       # 14GB — needs most of the VRAM
            "ip-adapter-faceid",  # 2GB
            "insightface",        # 0.5GB
            "realesrgan",         # 0.5GB
        ]

        for name in portrait_models:
            slot = self.models.get(name)
            if not slot:
                continue
            if self.gpu_available and remaining >= slot.size_gb:
                vram_models.append(name)
                remaining -= slot.size_gb

        return {
            "pipeline": "portrait_generation",
            "load_to_vram": vram_models,
            "unload_first": ["liveportrait", "whisper-large-v3-turbo"],
            "reload_after": ["liveportrait", "kokoro-tts", "silero-vad"],
            "estimated_vram_gb": self.vram_budget_gb - remaining,
            "note": "Portrait gen happens BEFORE conversation starts. "
                    "FLUX loads, generates, then unloads. Conversation models reload.",
        }

    def _plan_knowledge(self) -> dict:
        """Knowledge ingestion (Scholar). Whisper-heavy, no animation needed."""
        return {
            "pipeline": "knowledge_ingestion",
            "load_to_vram": ["whisper-large-v3-turbo", "minilm-embeddings"],
            "unload_first": ["liveportrait"],
            "reload_after": ["liveportrait"],
            "estimated_vram_gb": 3.1,
            "note": "Unload LivePortrait during bulk ingestion. "
                    "Whisper needs VRAM for fast transcription.",
        }

    def _plan_cinematic(self) -> dict:
        """Cinematic render (Path B). HunyuanVideo-Avatar needs 16GB+."""
        return {
            "pipeline": "cinematic",
            "load_to_vram": ["hunyuan-avatar"],
            "unload_first": ["liveportrait", "whisper-large-v3-turbo", "kokoro-tts"],
            "reload_after": ["liveportrait", "kokoro-tts", "silero-vad"],
            "estimated_vram_gb": 16.0,
            "note": "Cinematic mode takes over the entire GPU. "
                    "Not available during real-time conversation.",
        }

    # ── Model Loading / Offloading ─────────────────────────────────

    async def ensure_model_local(self, model_name: str) -> str:
        """
        Ensure a model is available locally (Seagate or SSD).
        Downloads from HuggingFace if not cached.
        Returns local path.
        """
        slot = self.models.get(model_name)
        if not slot:
            raise ValueError(f"Unknown model: {model_name}")

        # Already local
        if slot.current_tier in (StorageTier.LOCAL, StorageTier.RAM, StorageTier.VRAM):
            if slot.local_path:
                return slot.local_path

        # Determine download target
        if self.seagate_available:
            target_dir = Path(self.SEAGATE_CACHE) / model_name
            logger.info(f"Downloading {model_name} to Seagate: {target_dir}")
        else:
            target_dir = Path(self.LOCAL_CACHE) / model_name
            logger.info(f"Downloading {model_name} to local SSD: {target_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
            await asyncio.to_thread(
                snapshot_download,
                repo_id=slot.repo_id,
                local_dir=str(target_dir),
                token=os.getenv("HF_TOKEN"),
            )
            slot.current_tier = StorageTier.LOCAL
            slot.local_path = str(target_dir)
            logger.info(f"Downloaded {model_name} ({slot.size_gb}GB) to {target_dir}")
            return str(target_dir)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise

    async def sync_to_seagate(self) -> dict:
        """
        Sync all locally cached models to Seagate 5TB for fast access.
        Run this when Seagate is connected to build the local CDN.
        """
        if not self.seagate_available:
            return {"error": "Seagate 5TB not connected"}

        synced = []
        skipped = []
        errors = []

        for name, slot in self.models.items():
            seagate_path = Path(self.SEAGATE_CACHE) / name

            # Already on Seagate
            if slot.local_path and self.SEAGATE_CACHE in slot.local_path:
                skipped.append(name)
                continue

            # Copy from local SSD to Seagate
            local_path = Path(self.LOCAL_CACHE) / name
            if local_path.exists() and any(local_path.iterdir()):
                try:
                    if seagate_path.exists():
                        shutil.rmtree(str(seagate_path))
                    shutil.copytree(str(local_path), str(seagate_path))
                    slot.local_path = str(seagate_path)
                    synced.append(name)
                    logger.info(f"Synced {name} to Seagate")
                except Exception as e:
                    errors.append(f"{name}: {e}")

        return {
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
            "seagate_path": self.SEAGATE_CACHE,
        }

    async def upload_to_hf(self, model_name: str) -> str:
        """
        Upload a locally cached model to HuggingFace as backup.
        Uses AIBRUH namespace for the eden-os model cache repo.
        """
        slot = self.models.get(model_name)
        if not slot or not slot.local_path:
            raise ValueError(f"Model {model_name} not available locally")

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            # Use a dedicated cache repo on HF
            cache_repo = f"{self.HF_NAMESPACE}/eden-os-model-cache"

            # Create repo if it doesn't exist
            try:
                api.create_repo(cache_repo, repo_type="model", exist_ok=True)
            except Exception:
                pass

            # Upload the model directory
            api.upload_folder(
                folder_path=slot.local_path,
                repo_id=cache_repo,
                path_in_repo=model_name,
                token=os.getenv("HF_TOKEN"),
            )

            logger.info(f"Uploaded {model_name} to HF: {cache_repo}/{model_name}")
            return f"https://huggingface.co/{cache_repo}/tree/main/{model_name}"
        except Exception as e:
            logger.error(f"HF upload failed for {model_name}: {e}")
            raise

    # ── VRAM Swap Operations ───────────────────────────────────────

    async def load_to_vram(self, model_name: str) -> bool:
        """Load a model into GPU VRAM."""
        slot = self.models.get(model_name)
        if not slot:
            return False

        if not self.gpu_available:
            logger.debug(f"No GPU — {model_name} stays in RAM/CPU")
            slot.current_tier = StorageTier.RAM
            return True

        # Check VRAM budget
        if self.vram_used_gb + slot.size_gb > self.vram_budget_gb:
            # Need to evict something
            evicted = await self._evict_for_space(slot.size_gb)
            if not evicted:
                logger.warning(
                    f"Cannot load {model_name} ({slot.size_gb}GB) — "
                    f"VRAM full ({self.vram_used_gb:.1f}/{self.vram_budget_gb:.1f}GB)"
                )
                return False

        slot.vram_loaded = True
        slot.current_tier = StorageTier.VRAM
        slot.last_used = time.monotonic()
        self.vram_used_gb += slot.size_gb
        logger.info(
            f"Loaded {model_name} to VRAM "
            f"({self.vram_used_gb:.1f}/{self.vram_budget_gb:.1f}GB used)"
        )
        return True

    async def unload_from_vram(self, model_name: str) -> bool:
        """Offload a model from GPU VRAM back to RAM."""
        slot = self.models.get(model_name)
        if not slot or not slot.vram_loaded:
            return False

        slot.vram_loaded = False
        slot.current_tier = StorageTier.RAM
        self.vram_used_gb = max(0, self.vram_used_gb - slot.size_gb)
        logger.info(
            f"Offloaded {model_name} from VRAM "
            f"({self.vram_used_gb:.1f}/{self.vram_budget_gb:.1f}GB used)"
        )

        # Trigger garbage collection to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return True

    async def _evict_for_space(self, needed_gb: float) -> bool:
        """Evict lowest-priority models from VRAM to make space."""
        # Sort loaded models by priority (evict LOW first) then by last_used (LRU)
        loaded = [
            (name, slot) for name, slot in self.models.items()
            if slot.vram_loaded
        ]
        loaded.sort(key=lambda x: (-x[1].priority.value, x[1].last_used))

        freed = 0.0
        for name, slot in loaded:
            if slot.priority == ModelPriority.CRITICAL:
                continue  # Never evict CRITICAL models

            await self.unload_from_vram(name)
            freed += slot.size_gb
            if freed >= needed_gb:
                return True

        return freed >= needed_gb

    # ── Prefetch / Background Optimization ─────────────────────────

    async def prefetch_pipeline_models(self, pipeline_type: str = "conversation") -> dict:
        """
        Pre-download all models needed for a pipeline to local storage.
        Run this during setup to eliminate download latency during conversation.
        """
        plan = self.get_vram_plan(pipeline_type)
        all_needed = plan["load_to_vram"] + plan.get("keep_in_ram", [])

        results = {"downloaded": [], "already_local": [], "failed": []}

        for model_name in all_needed:
            slot = self.models.get(model_name)
            if not slot:
                continue

            if slot.current_tier in (StorageTier.LOCAL, StorageTier.RAM, StorageTier.VRAM):
                results["already_local"].append(model_name)
                continue

            try:
                await self.ensure_model_local(model_name)
                results["downloaded"].append(model_name)
            except Exception as e:
                results["failed"].append(f"{model_name}: {e}")

        return results

    # ── Status / Reporting ─────────────────────────────────────────

    def get_storage_report(self) -> dict:
        """Get comprehensive storage status across all tiers."""
        tier_summary = {tier.value: [] for tier in StorageTier}

        for name, slot in self.models.items():
            tier_summary[slot.current_tier.value].append({
                "name": name,
                "size_gb": slot.size_gb,
                "priority": slot.priority.name,
                "engine": slot.engine,
            })

        # Calculate totals per tier
        tier_totals = {}
        for tier, models in tier_summary.items():
            tier_totals[tier] = {
                "model_count": len(models),
                "total_gb": sum(m["size_gb"] for m in models),
                "models": models,
            }

        # Storage availability
        storage = {
            "gpu": {
                "available": self.gpu_available,
                "budget_gb": self.vram_budget_gb,
                "used_gb": self.vram_used_gb,
                "free_gb": self.vram_budget_gb - self.vram_used_gb,
            },
            "seagate": {
                "available": self.seagate_available,
                "mount": self.SEAGATE_MOUNT,
                "cache_path": self.SEAGATE_CACHE,
            },
            "local_ssd": {
                "cache_path": self.LOCAL_CACHE,
            },
            "huggingface": {
                "namespace": self.HF_NAMESPACE,
                "storage": "1TB",
            },
        }

        # Check actual Seagate free space
        if self.seagate_available:
            try:
                stat = os.statvfs(self.SEAGATE_MOUNT)
                storage["seagate"]["total_gb"] = round(
                    (stat.f_blocks * stat.f_frsize) / (1024 ** 3), 1
                )
                storage["seagate"]["free_gb"] = round(
                    (stat.f_bavail * stat.f_frsize) / (1024 ** 3), 1
                )
            except OSError:
                pass

        return {
            "tiers": tier_totals,
            "storage": storage,
            "total_model_size_gb": sum(s.size_gb for s in self.models.values()),
        }

    def get_swap_recommendation(self) -> str:
        """
        Get a human-readable recommendation for current hardware.
        """
        total = sum(s.size_gb for s in self.models.values())

        if self.vram_budget_gb >= 80:
            return (
                f"H100/A100 ({self.vram_budget_gb}GB): Load everything. "
                f"Run both Path A + Path B simultaneously. Cinema mode available."
            )
        elif self.vram_budget_gb >= 24:
            return (
                f"RTX 4090/3090 ({self.vram_budget_gb}GB): Conversation pipeline fits "
                f"comfortably (~10GB). Swap in FLUX for portrait gen. "
                f"Keep {total:.0f}GB of models on Seagate for instant swaps."
            )
        elif self.vram_budget_gb > 0:
            return (
                f"GPU ({self.vram_budget_gb}GB): Tight fit. LivePortrait + Kokoro only. "
                f"Whisper loads on demand. FLUX requires full VRAM swap."
            )
        else:
            return (
                f"CPU-only mode: All models run on CPU. Seagate as model cache. "
                f"Reduced FPS but functional. {total:.0f}GB total models — "
                f"Seagate 5TB has plenty of room. HF Hub as backup."
            )

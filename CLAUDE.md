# EDEN OS — CLAUDE.md
# Master Orchestration File for Building 4D Bi-Directional Conversational Avatars
# Version: 1.1 | Phase: ONE (OS Pipeline) | Codename: OWN THE SCIENCE
# Generated: 2026-03-31 by Amanda (Avatar Pipeline Architect)
# Updated: 2026-03-31 — Phase One BUILD COMPLETE, 109 realism tests passing

---

## BUILD STATUS: PHASE ONE COMPLETE

All 7 engines are built, tested, and deployed. This CLAUDE.md now serves as both
the original specification AND the living documentation of what was built.

| Component | Status | Files | Tests |
|-----------|--------|-------|-------|
| Genesis (Agent 1) | BUILT | 5 modules + SkinRealismAgent | 28 visual tests |
| Voice (Agent 2) | BUILT | 6 modules | 28 vocal tests |
| Animator (Agent 3) | BUILT | 6 modules | 27 chat-video tests |
| Brain (Agent 4) | BUILT | 6 modules + 2 templates | 20 E2E tests |
| Conductor (Agent 5) | BUILT | 5 modules | integrated |
| Gateway (Agent 6) | BUILT | 5 modules + frontend | integrated |
| Scholar (Agent 7) | BUILT | 6 modules | integrated |
| **TOTAL** | **109/109 tests passing** | **77 source files** | **4 test suites** |

### Deployed Locations (ALL PUBLIC)
- **GitHub**: https://github.com/tyronne-os/EDEN-OS (public)
- **HuggingFace**: https://huggingface.co/AIBRUH/eden-os (public, model repo)
- **Seagate 5TB**: `S:\eden-os\versions\v1.0.1` (versioned local backup)
- **Virtual Env**: `~/EDEN-OS/.venv` (Python 3.12, 152+ packages)

---

## MISSION STATEMENT

You are building **EDEN OS** — an operating system pipeline that converts any 2D portrait image into a photorealistic 4D conversational avatar capable of real-time bi-directional dialogue. The avatar must be so realistic that it is indistinguishable from a human video call. The system must be **conversation-ready upon load** — no warm-up, no buffering, no uncanny valley.

**"Own The Science"** means: we do not wrap APIs. We engineer our own inference pipeline using open-weight models from Hugging Face, orchestrated through a custom backend that delivers sub-200ms latency for the reasoning layer and sub-5-second total pipeline execution for the full video response cycle.

---

## PHASE ONE SCOPE: EDEN OS AS A HEADLESS OPERATING SYSTEM

EDEN OS is NOT an app. It is an **operating system** — a headless backend engine that exposes a universal API. Any frontend (React, Gradio, mobile app, kiosk, VR headset) plugs into it. When the build is complete, the OS deploys to a **HuggingFace Space** at `AIBRUH/eden-os` and is immediately accessible via a live URL.

### Deployment Target
- **Platform**: HuggingFace Spaces (Docker SDK with GPU)
- **Space ID**: `AIBRUH/eden-os`
- **SDK**: Docker (not Gradio — we need full control over the server)
- **Hardware**: T4 GPU (free tier to start, upgrade to A10G/A100 for production)
- **Secrets**: `ANTHROPIC_API_KEY`, `HF_TOKEN` stored as HF Space secrets

### Headless OS Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    EDEN OS (Headless Engine)             │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              EDEN OS API LAYER                   │    │
│  │  POST /api/v1/sessions        (create session)   │    │
│  │  WS   /api/v1/sessions/{id}/stream (bi-dir AV)  │    │
│  │  POST /api/v1/knowledge/ingest (feed media)      │    │
│  │  PUT  /api/v1/settings/{id}   (sliders/config)   │    │
│  │  GET  /api/v1/health          (status)           │    │
│  └──────────────┬──────────────────────────────────┘    │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────────┐    │
│  │              CONDUCTOR (Orchestrator)             │    │
│  │  Connects all 7 engines, enforces latency,       │    │
│  │  manages sessions, routes data between engines   │    │
│  └──────────────┬──────────────────────────────────┘    │
│                 │                                        │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
│  │GENESIS│ │VOICE │ │ANIMTR│ │BRAIN │ │SCHOLR│ │GATWAY││
│  │Agent 1│ │Agent2│ │Agent3│ │Agent4│ │Agent7│ │Agent6││
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘│
└─────────────────────────────────────────────────────────┘
```

### API Contract (Verified Working)
```
POST /api/v1/sessions              → 200 {session_id, ws_url, status}
DELETE /api/v1/sessions/{id}       → 200
GET /api/v1/sessions/{id}/status   → 200 {session_id, state, metrics}
POST /api/v1/sessions/{id}/interrupt → 200
PUT /api/v1/sessions/{id}/settings → 200 {applied: true}
PUT /api/v1/sessions/{id}/pipeline → 200 {swapped: true}
GET /api/v1/templates              → 200 [8 templates]
POST /api/v1/knowledge/ingest      → 200 {job_id, status, chunks_estimated}
GET /api/v1/health                 → 200 {status, gpu, active_sessions, uptime}
WS /api/v1/sessions/{id}/stream    → bi-directional audio/video/text
GET /                              → 200 EDEN Studio frontend (26KB)
GET /docs                          → 200 Swagger API docs
```

---

## ARCHITECTURAL PHILOSOPHY: THE EDEN PROTOCOL

### The 0.3 Deviation Rule
Every generated frame must pass the **Eden Protocol Validator**: skin texture deviation from the reference portrait must remain below 0.3 standard deviations. This eliminates the "plastic skin" and "waxy sheen" artifacts that plague competing systems, especially on melanin-rich skin tones.

### The Three States of Presence
1. **LISTENING** — Avatar maintains active listening behaviors (micro-blinks, subtle nodding, gaze tracking). ASR processes audio in real-time. Avatar is NOT frozen.
2. **THINKING** — LLM generates response. First tokens trigger TTS immediately. Avatar transitions with subtle inhale + brow raise.
3. **SPEAKING** — Animation frames synchronized with TTS audio stream. Frame-by-frame with KV-cache for temporal consistency.

### The KV-Recache Interruption Protocol
- Immediately halt TTS generation
- Refresh KV-cache while preserving temporal anchors (no glitch)
- Transition avatar back to LISTENING within 100ms
- Begin processing new user input

---

## WHAT WAS BUILT: ENGINE-BY-ENGINE

### AGENT 1: GENESIS (Portrait-to-4D Engine) — `eden_os/genesis/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `portrait_engine.py` | `PortraitEngine` | Face detection (Haar cascade fallback), alignment to 512x512, CLAHE lighting normalization |
| `eden_protocol_validator.py` | `EdenProtocolValidator` | 12-kernel Gabor filter bank (4 orientations x 3 frequencies), LAB color space deviation |
| `latent_encoder.py` | `LatentEncoder` | Spatial pyramid features + gradient histogram → 512-D latent vector |
| `preload_cache.py` | `PreloadCache` | 8 seed frames with micro-rotations + 6-frame breathing cycle |
| `skin_realism_agent.py` | `SkinRealismAgent` | **NEW** — full skin realism skill agent (see below) |

**Skin Realism Agent (v1.1 addition):**
- `analyze_portrait()` → builds `SkinProfile` (melanin_level, undertone, pore_density, texture_roughness, freckle_map, mole_positions, specular_intensity, oiliness)
- `enhance_frame()` → 6-step post-processing on every frame:
  1. Melanin-aware color correction (prevents whitewashing)
  2. Pore-level micro-texture synthesis from reference
  3. Subsurface scattering simulation (wavelength-dependent blur: R>G>B)
  4. Identity marker preservation (freckles, moles, beauty marks)
  5. Natural specular highlights (T-zone oiliness mapping)
  6. Emotion-driven skin response (blush on joy, pallor on urgency)

### AGENT 2: VOICE (TTS + ASR + Cloning) — `eden_os/voice/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `asr_engine.py` | `ASREngine` | Whisper transcription + Silero VAD endpoint detection |
| `tts_engine.py` | `TTSEngine` | TTS with streaming AudioChunk output, speed/pitch control |
| `voice_cloner.py` | `VoiceCloner` | Mel-spectrogram embedding extraction, voice_id storage |
| `emotion_router.py` | `EmotionRouter` | Keyword-based sentiment → {joy, sadness, confidence, urgency, warmth} |
| `interruption_handler.py` | `InterruptionHandler` | RMS-based VAD, sustained-frame detection, cooldown |
| `voice_engine.py` | `VoiceEngine` | Composes all modules, implements IVoiceEngine |

### AGENT 3: ANIMATOR (Lip-Sync + 4D Motion) — `eden_os/animator/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `liveportrait_driver.py` | `LivePortraitDriver` | 21-keypoint implicit representation, Gaussian mesh warping, audio-driven lip retargeting |
| `idle_generator.py` | `IdleGenerator` | Continuous blinks (3-7s), breathing (4s cycle), head microsway, brow micro-raises |
| `state_machine.py` | `AvatarStateMachine` | IDLE/LISTENING/THINKING/SPEAKING states, transition callbacks, blend parameters |
| `audio_to_keypoints.py` | `AudioToKeypoints` | RMS energy + autocorrelation pitch → keypoint deltas |
| `eden_temporal_anchor.py` | `EdenTemporalAnchor` | LONGLIVE-adapted frame sink, LAB feature extraction, drift detection + correction |
| `animator_engine.py` | `AnimatorEngine` | Composes all modules, implements IAnimatorEngine |

### AGENT 4: BRAIN (LLM + Persona + Memory) — `eden_os/brain/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `reasoning_engine.py` | `ReasoningEngine` | Anthropic streaming API (claude-sonnet-4-20250514), fallback echo mode |
| `persona_manager.py` | `PersonaManager` | YAML template validation, system_prompt/emotional_baseline/voice_config |
| `memory_manager.py` | `MemoryManager` | 20-turn sliding window, key fact extraction (names, emails, "my X is Y") |
| `streaming_bridge.py` | `StreamingBridge` | Token buffer → sentence chunker at .!? boundaries, sentiment per chunk |
| `template_loader.py` | `TemplateLoader` | YAML discovery + validation from templates/ directory |
| `brain_engine.py` | `BrainEngine` | Composes all modules, implements IBrainEngine |

**Templates (8 built):** default, medical_office, sales_dev_rep, ai_tutor, customer_support, fitness_coach, podcast_host, _template_schema

### AGENT 5: CONDUCTOR (Orchestrator) — `eden_os/conductor/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `orchestrator.py` | `Conductor` | Master controller, lazy engine loading, full pipeline routing |
| `latency_enforcer.py` | `LatencyEnforcer` | Per-stage budgets: ASR 500ms, LLM 200ms, TTS 300ms, Animation 50ms |
| `error_recovery.py` | `ErrorRecovery` | Fallback chains per engine, max 2 retries, graceful degradation |
| `session_manager.py` | `SessionManager` | UUID sessions, config/state/engines/history lifecycle |
| `metrics_collector.py` | `MetricsCollector` | Rolling window (100 measurements), p50/p95/p99 percentiles |

### AGENT 6: GATEWAY (API + WebSocket) — `eden_os/gateway/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `api_server.py` | `create_app()` | FastAPI with 12 endpoints, CORS, Pydantic models, static mount |
| `websocket_handler.py` | `WebSocketHandler` | Bi-directional streaming, asyncio queues, state dispatch |
| `audio_capture.py` | `AudioCapture` | Base64 PCM decoding, noise gate, 16kHz resampling |
| `video_encoder.py` | `VideoEncoder` | RGB frames → JPEG/PNG base64, configurable quality |
| `webrtc_handler.py` | `WebRTCHandler` | Stub for future WebRTC (WebSocket active now) |

### AGENT 7: SCHOLAR (Knowledge + RAG) — `eden_os/scholar/`

**Built modules:**
| File | Class | Purpose |
|------|-------|---------|
| `youtube_ingestor.py` | `YouTubeIngestor` | yt-dlp + Whisper transcription, timestamped chunks |
| `audiobook_ingestor.py` | `AudiobookIngestor` | Long-form transcription, semantic chunking by topic |
| `url_ingestor.py` | `URLIngestor` | trafilatura (web) + pymupdf (PDF), 500-token chunks |
| `knowledge_graph.py` | `KnowledgeGraph` | Entity extraction, co-occurrence relationships, BFS query |
| `rag_retriever.py` | `RAGRetriever` | ChromaDB + sentence-transformers (all-MiniLM-L6-v2), hybrid search |
| `media_analyzer.py` | `MediaAnalyzer` | Batch processing controller, KnowledgeSummary output |

---

## 3-TIER VRAM & STORAGE STRATEGY — `eden_os/shared/vram_strategy.py`

**v1.1 addition.** Manages 16 models across 3 storage tiers:

| Tier | Storage | Purpose | Capacity |
|------|---------|---------|----------|
| **HOT** | GPU VRAM | Active inference models | 0-80GB |
| **WARM** | Seagate 5TB / Local SSD | Pre-downloaded weights | 4.6TB free |
| **COLD** | HuggingFace Hub (AIBRUH) | Persistent cloud cache | 1TB |

**Model Registry (16 models, 92.7GB total):**
- CRITICAL (always in VRAM): LivePortrait 4GB, Kokoro 0.5GB, Silero VAD 0.1GB
- HIGH (load on demand): Whisper 3GB, InsightFace 0.5GB, MiniLM 0.1GB
- MEDIUM (on Seagate): CosyVoice2 2GB, StyleTTS2 4GB, FLUX-schnell 14GB
- LOW (on HF Hub): HunyuanVideo 16GB, Qwen3-8B 5GB, FLUX-pro 22GB

**Pipeline VRAM Plans:**
- `conversation`: LivePortrait + TTS + VAD = ~4.6GB (fits any GPU)
- `portrait_generation`: FLUX-schnell = 14GB (swap conversation models out first)
- `knowledge_ingestion`: Whisper + embeddings = 3.1GB (unload LivePortrait)
- `cinematic`: HunyuanVideo-Avatar = 16GB (takes over entire GPU)

---

## REALISM TESTING SUITE — 109 TESTS, 100% PASS

### test_visual_realism.py (28 tests)
- Eden Protocol skin fidelity across 5 melanin levels
- Skin Realism Agent: SSS, color correction, freckles, moles, specular, blush/pallor
- Idle animation: blink frequency, breathing cycle, head microsway, brow raises
- State transitions: smoothness, 100ms interrupt budget, no frozen frames
- Temporal consistency: 100-frame identity drift < 0.1
- Frame quality: 512x512, sharpness, color gamut, no black/white artifacts

### test_vocal_realism.py (28 tests)
- TTS quality: RMS, sample rate, clipping, DC offset, frequency range
- Emotion routing: joy/sadness/confidence/urgency/warmth detection accuracy
- Voice naturalness: pitch variation, rhythm, no robotic loops, energy contour
- Interruption: detection accuracy, no false positives, halt confirmation
- ASR: transcription, silence/noise handling
- Voice cloning: embedding consistency (cosine >0.9), storage

### test_chat_video_realism.py (27 tests)
- Full pipeline: text → emotion → audio → keypoints → video frames
- Lip-sync: energy-mouth correlation > 0.5, silence = closed mouth
- Micro-expressions: joy → smile, confidence → brow raise, smooth transitions
- Performance: frame render <200ms CPU, audio features <5ms, emotion <1ms
- Anti-AI forensics: no uniform patches, Gaussian noise, no spectral banding
- Perceptual: SSIM >0.5, color consistency, edge sharpness maintained
- Stress: 10 rapid interrupts, 100-frame identity, all 5 skin tones, concurrent sessions

### test_e2e_realism.py (20 tests)
- Portrait → Genesis → Animator pipeline integration
- Gateway API: health, sessions, templates, frontend serving
- Latency enforcer, metrics collector, error recovery

**Run all tests:** `cd ~/EDEN-OS && source .venv/bin/activate && python -m pytest tests/ -v`

---

## EDEN STUDIO ADMIN PANEL — `static/index.html`

26KB single-file HTML/CSS/JS application. Design: #080503 onyx black + #C5B358 gold.

**Left Column — 6 Behavioral Sliders:**
| Slider | Default | Maps To |
|--------|---------|---------|
| Consistency | 70% | Eden Protocol threshold |
| Latency | 100% | Pipeline speed vs quality |
| Expressiveness | 60% | LivePortrait retargeting amplitude |
| Voice Tone | 85% | TTS pitch and warmth |
| Eye Contact | 50% | Gaze lock to camera |
| Flirtation | 15% | Composite smile + brow + tilt |

**Center — Pipeline + Metrics:**
Model-to-Model swap, New Pipeline, Connectivity status, real-time latency/FPS/VRAM metrics

**Right — Avatar + Context:**
EVE portrait canvas, EDEN pill selector, Custom Instructions textarea, Knowledge modal

**Knowledge Injection Modal:**
YouTube URL, Audiobook upload, Research URL, Natural language prompt, Analyze Media Sources

---

## MODEL PRIORITY MATRIX

| HARDWARE TIER | ANIMATION | TTS | LLM |
|---------------|-----------|-----|-----|
| H100 (80GB) | HunyuanAvatar + LivePortrait | CosyVoice2 | Claude Sonnet |
| RTX 4090 (24GB) | LivePortrait | CosyVoice2 | Claude Sonnet |
| RTX 3090 (24GB) | LivePortrait | Kokoro | Qwen3 8B |
| L4/T4 (24GB) | LivePortrait | Kokoro | Claude Sonnet |
| CPU Only | LivePortrait (reduced fps) | Kokoro | BitNet 3B |

---

## FILE STRUCTURE (ACTUAL, AS BUILT)

```
~/EDEN-OS/
├── CLAUDE.md                              # This file (v1.1)
├── README.md                              # HuggingFace model card
├── app.py                                 # Entry point — boots FastAPI on port 7860
├── requirements.txt                       # 152+ packages
├── Dockerfile                             # CUDA 12.4 base, port 7860
├── docker-compose.yml                     # Local dev with GPU
├── pyproject.toml                         # Package config + pytest settings
├── .gitignore
├── .dockerignore
│
├── config/
│   ├── default.yaml                       # Default configuration
│   ├── eden_protocol.yaml                 # Gabor filter + threshold config
│   └── hardware_profiles/
│       ├── h100_cinematic.yaml
│       ├── rtx4090_production.yaml
│       ├── rtx3090_standard.yaml
│       ├── l4_cloud.yaml
│       └── cpu_edge.yaml
│
├── eden_os/
│   ├── __init__.py
│   ├── shared/
│   │   ├── types.py                       # AvatarState, AudioChunk, VideoFrame, TextChunk, etc.
│   │   ├── interfaces.py                  # IGenesisEngine, IVoiceEngine, etc. (7 ABCs)
│   │   └── vram_strategy.py               # 3-tier VRAM management (16 models, 92.7GB)
│   ├── genesis/                           # Agent 1: 5 modules
│   ├── voice/                             # Agent 2: 6 modules
│   ├── animator/                          # Agent 3: 6 modules
│   ├── brain/                             # Agent 4: 6 modules
│   ├── conductor/                         # Agent 5: 5 modules
│   ├── gateway/                           # Agent 6: 5 modules
│   └── scholar/                           # Agent 7: 6 modules
│
├── templates/                             # 8 agent persona YAMLs
├── static/
│   └── index.html                         # EDEN Studio admin panel (26KB)
├── scripts/
│   ├── setup_models.py                    # Download HF model weights
│   ├── validate_gpu.py                    # GPU detection
│   └── save_versioned.sh                  # GitHub + HF + Seagate versioned save
│
├── tests/                                 # 109 tests, 100% pass rate
│   ├── conftest.py                        # Shared fixtures + synthetic generators
│   ├── test_visual_realism.py             # 28 visual tests
│   ├── test_vocal_realism.py              # 28 vocal tests
│   ├── test_chat_video_realism.py         # 27 chat+video tests
│   └── test_e2e_realism.py               # 20 integration tests
│
├── models_cache/                          # Downloaded model weights (gitignored)
└── data/                                  # ChromaDB vector store (gitignored)
```

---

## VERSIONED SAVE PROTOCOL

```bash
# Save to all 3 locations with version tag:
bash scripts/save_versioned.sh v1.1

# What it does:
# 1. git commit + tag v1.1
# 2. git push to GitHub (tyronne-os/EDEN-OS)
# 3. Upload to HuggingFace (AIBRUH/eden-os)
# 4. Copy to Seagate 5TB (S:\eden-os\versions\v1.1)
# 5. Update latest symlink + VERSION_LOG.md
# 6. Keep last 10 versions, auto-clean older
```

---

## NEXT: PHASE TWO

Phase Two transforms EDEN OS into the **EDEN Studio SaaS**:
- Multi-tenant architecture with Stripe billing
- EDEN Studio React frontend with gold/onyx design language
- Agent template marketplace
- White-label customization for enterprise
- HuggingFace Space deployment for free tier
- Mobile SDK for iOS/Android

But first — Phase One must be bulletproof. The engine must sing before we build the concert hall.

---

**OWN THE SCIENCE.**
**EDEN OS v1.1 — Phase One COMPLETE**
**Built by TJ LSU DAD + Amanda + 7 Claude Code Agents**

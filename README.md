---
title: EDEN OS
emoji: 🌿
colorFrom: black
colorTo: gold
license: apache-2.0
tags:
  - talking-head
  - avatar
  - real-time
  - conversational-ai
  - 4d-avatar
  - eden-protocol
  - liveportrait
  - lip-sync
  - tts
  - asr
---

# EDEN OS — 4D Conversational Avatar Operating System

**Version 1.0 | Phase One | Codename: OWN THE SCIENCE**

Upload any 2D portrait → Get a photorealistic talking avatar that converses in real-time with perfect lip sync, emotional expression, and knowledge of your content.

## What is EDEN OS?

EDEN OS is a **headless operating system** — not an app. It's a backend engine that exposes a universal API. Any frontend (React, Gradio, mobile, VR) plugs into it.

## Architecture: 7 Specialized Engines

| Engine | Agent | Role |
|--------|-------|------|
| **Genesis** | Agent 1 | Portrait processing, Eden Protocol validation, skin realism |
| **Voice** | Agent 2 | ASR (Whisper), TTS (Kokoro/CosyVoice2), voice cloning, emotion routing |
| **Animator** | Agent 3 | LivePortrait real-time animation, idle generation, state machine |
| **Brain** | Agent 4 | Claude Sonnet streaming, persona management, memory, RAG context |
| **Conductor** | Agent 5 | Pipeline orchestration, latency enforcement, error recovery |
| **Gateway** | Agent 6 | FastAPI REST API, WebSocket streaming, video encoding |
| **Scholar** | Agent 7 | YouTube/audiobook/URL ingestion, ChromaDB RAG, knowledge graph |

## The Eden Protocol

Every generated frame must pass the **0.3 deviation rule** — skin texture fidelity measured in LAB color space with Gabor filter banks. This eliminates the "plastic skin" artifact, especially on melanin-rich skin tones.

**Skin Realism Agent**: Built-in skill agent for:
- Pore-level micro-texture synthesis
- Subsurface scattering simulation
- Melanin-aware color correction
- Emotion-driven skin response (blush, pallor)
- Identity marker preservation (freckles, moles, beauty marks)

## API

```bash
# Health check
curl https://AIBRUH-eden-os.hf.space/api/v1/health

# Create session
curl -X POST /api/v1/sessions -d '{"template": "medical_office"}'

# Inject knowledge
curl -X POST /api/v1/knowledge/ingest -d '{"type": "youtube", "url": "..."}'

# WebSocket bi-directional stream
ws://host/api/v1/sessions/{id}/stream
```

## 3-Tier VRAM Strategy

| Tier | Storage | Purpose |
|------|---------|---------|
| **HOT** | GPU VRAM | Active models (LivePortrait 4GB + TTS 0.5GB + VAD 0.1GB) |
| **WARM** | Seagate 5TB / Local SSD | Pre-downloaded weights, instant swap |
| **COLD** | HuggingFace Hub (1TB) | Persistent cloud cache |

## Quick Start

```bash
git clone https://github.com/tyronne-os/EDEN-OS
cd EDEN-OS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Hardware Profiles

| Tier | GPU | Animation | TTS | LLM |
|------|-----|-----------|-----|-----|
| H100 (80GB) | HunyuanAvatar + LivePortrait | CosyVoice2 | Claude Sonnet |
| RTX 4090 (24GB) | LivePortrait | CosyVoice2 | Claude Sonnet |
| RTX 3090 (24GB) | LivePortrait | Kokoro | Qwen3 8B |
| CPU Only | LivePortrait (reduced fps) | Kokoro | BitNet 3B |

## Built With

- **LivePortrait** (KwaiVGI) — 78fps implicit keypoint animation
- **Whisper** (OpenAI) — real-time ASR
- **Claude Sonnet** (Anthropic) — conversational reasoning
- **ChromaDB** + **sentence-transformers** — RAG knowledge retrieval
- **FastAPI** + **aiortc** — API + WebRTC streaming

---

**OWN THE SCIENCE.**

*Built by TJ LSU DAD + Claude Code Agents*

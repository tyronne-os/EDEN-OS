# EDEN OS — CLAUDE.md
# Master Orchestration File for Building 4D Bi-Directional Conversational Avatars
# Version: 1.0 | Phase: ONE (OS Pipeline) | Codename: OWN THE SCIENCE
# Generated: 2026-03-31 by Amanda (Avatar Pipeline Architect)

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
- **Live URL**: `https://huggingface.co/spaces/AIBRUH/eden-os`
- **SDK**: Docker (not Gradio — we need full control over the server)
- **Hardware**: T4 GPU (free tier to start, upgrade to A10G/A100 for production)
- **Secrets**: `ANTHROPIC_API_KEY`, `HF_TOKEN` stored as HF Space secrets

### What the URL delivers
When you click the link, you see the **EDEN Studio admin panel** — EVE is displayed, idle-animating (blinking, breathing), ready to converse. The full admin panel UI loads: behavioral sliders, knowledge injection modal, pipeline controls. Click "Initiate Conversation" and EVE is live.

### Headless OS Architecture (Scalability from Ground Zero)
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
         │                                    ▲
         ▼                                    │
┌─────────────────────────────────────────────────────────┐
│              ANY FRONTEND (Pluggable)                    │
│                                                         │
│  Option A: EDEN Studio (built-in, served at / )         │
│  Option B: React/Next.js SaaS app (Phase Two)           │
│  Option C: Gradio interface (rapid prototyping)          │
│  Option D: Mobile SDK (iOS/Android)                     │
│  Option E: Third-party integration via API               │
│  Option F: Embed widget (like Naoma's website embed)     │
└─────────────────────────────────────────────────────────┘
```

### The Core Principle
Every feature in EDEN OS is accessible through the API. The built-in EDEN Studio frontend is just one client. A developer should be able to `curl` the API and get a talking avatar response. This is what makes it an OS, not an app.

### API Contract (Universal Interface)
```
# Create a session — returns session_id + WebSocket URL
POST /api/v1/sessions
  Body: { portrait_image: base64, template: "medical_office" }
  Returns: { session_id, ws_url, status: "ready" }

# Stream bi-directional conversation (WebSocket)
WS /api/v1/sessions/{id}/stream
  Send: { type: "audio", data: base64_pcm }     ← user speaks
  Send: { type: "text", content: "hello" }       ← or types
  Send: { type: "interrupt" }                    ← user interrupts
  Receive: { type: "video_frame", data: base64 } ← animated avatar frame
  Receive: { type: "audio", data: base64_wav }   ← avatar voice
  Receive: { type: "transcript", text: "..." }   ← what avatar said
  Receive: { type: "state", value: "speaking" }  ← current state

# Inject knowledge (YouTube, audiobook, URL)
POST /api/v1/knowledge/ingest
  Body: { type: "youtube", url: "https://..." }
  Body: { type: "audiobook", file: base64_mp3 }
  Body: { type: "url", url: "https://arxiv.org/..." }
  Returns: { job_id, status: "processing", chunks_estimated: 127 }

# Update behavioral settings in real-time
PUT /api/v1/sessions/{id}/settings
  Body: { expressiveness: 0.8, eye_contact: 1.0, voice_tone: 0.85 }
  Returns: { applied: true }

# Swap models mid-session
PUT /api/v1/sessions/{id}/pipeline
  Body: { tts_engine: "styletts2", animation_engine: "hunyuan" }
  Returns: { swapped: true, reload_time_ms: 2400 }
```

### HuggingFace Space Structure (Dockerfile-based)
```
AIBRUH/eden-os/
├── Dockerfile                    # Multi-stage build, CUDA base image
├── app.py                        # Entry point — boots FastAPI + all engines
├── requirements.txt
├── static/
│   └── index.html                # EDEN Studio admin panel (built-in frontend)
├── eden_os/
│   ├── __init__.py
│   ├── genesis/                  # Agent 1
│   ├── voice/                    # Agent 2
│   ├── animator/                 # Agent 3
│   ├── brain/                    # Agent 4
│   ├── conductor/                # Agent 5
│   ├── gateway/                  # Agent 6
│   ├── scholar/                  # Agent 7
│   └── shared/                   # Shared types, interfaces, config
├── templates/                    # Agent persona YAMLs
├── models_cache/                 # Downloaded HF model weights (persistent volume)
└── README.md                     # HF Space card
```

### Existing AIBRUH Spaces Integration
Your existing HuggingFace Spaces become specialized microservices that EDEN OS can call:
- `AIBRUH/eve-voice-engine` → Voice engine can delegate to this for advanced TTS
- `AIBRUH/eden-realism-engine` → Genesis can call this for Eden Protocol validation
- `AIBRUH/eden-diffusion-studio` → Genesis can call this for FLUX portrait generation
- `AIBRUH/eden-comfyui-pipeline` → Animator can call this for advanced ComfyUI workflows
- `AIBRUH/eden-video-studio` → Conductor can delegate cinematic renders here

The core `AIBRUH/eden-os` Space is the brain that orchestrates everything. The other Spaces become optional accelerators.

### Phase One deliverable:
1. `AIBRUH/eden-os` Space is live on HuggingFace
2. Clicking the URL opens EDEN Studio with EVE ready to converse
3. The API is accessible at `https://AIBRUH-eden-os.hf.space/api/v1/`
4. Any developer can integrate EDEN OS into their own frontend via the API

---

## ARCHITECTURAL PHILOSOPHY: THE EDEN PROTOCOL

### The 0.3 Deviation Rule
Every generated frame must pass the **Eden Protocol Validator**: skin texture deviation from the reference portrait must remain below 0.3 standard deviations. This eliminates the "plastic skin" and "waxy sheen" artifacts that plague competing systems, especially on melanin-rich skin tones. This is our signature. This is how people know EDEN made it.

### The Three States of Presence
A truly bi-directional avatar must handle three simultaneous states:

1. **LISTENING** — The avatar maintains active listening behaviors (micro-blinks, subtle nodding, gaze tracking) while the user speaks. ASR processes audio in real-time. The avatar is NOT frozen.
2. **THINKING** — The LLM generates a response. First tokens trigger TTS immediately (streaming response). The avatar transitions from listening to a "processing" micro-expression (slight brow raise, inhale).
3. **SPEAKING** — The 4D diffusion/animation model generates video frames synchronized with the TTS audio stream. Frame-by-frame autoregressive generation with KV-cache for temporal consistency.

### The KV-Recache Interruption Protocol
When a user interrupts mid-response, the system must:
- Immediately halt TTS generation
- Refresh the KV-cache for future frames while preserving temporal anchors of the current face position (no glitch, no jump-cut)
- Transition the avatar back to LISTENING state within 100ms
- Begin processing the new user input

This is adapted from the LONGLIVE framework (arXiv:2509.22622) and is what separates EDEN from every "talking head" on the market.

---

## EDEN STUDIO ADMIN PANEL SPECIFICATION

The admin panel is the operator's control surface for EDEN OS. It has two views derived from the prototype UI: the **Main Control Surface** and the **Knowledge Injection Modal**. Every element maps to a real backend function.

### Main Control Surface (Admin Panel 1)

**Layout**: Three-column design on black (#080503) background. Left column: Settings + Backend. Center column: Pipeline controls + Connectivity. Right column: EVE avatar + context editor.

**Left Column — Behavioral Sliders**:
These sliders control LivePortrait's retargeting parameters and the Voice/Brain engines in real-time:

| Slider | Default | Maps To | Engine |
|--------|---------|---------|--------|
| **Consistency** | ~70% | Eden Protocol threshold. 100% = strict 0.3 deviation. Lower = relaxed matching | Genesis → `eden_protocol_validator.py` |
| **Latency** | 100% | Pipeline priority. 100% = max speed (Schnell, Kokoro, skip upscale). 0% = max quality (FLUX Pro, StyleTTS2, full upscale) | Conductor → `latency_enforcer.py` |
| **Expressiveness** | ~60% | LivePortrait retargeting amplitude. High = wide mouth, big brow raises. Low = subtle, reserved | Animator → `liveportrait_driver.py` expression_scale |
| **Voice Tone** | 85% | TTS pitch and warmth. High = warmer, richer. Low = neutral, clinical | Voice → `tts_engine.py` tone_warmth |
| **Eye Contact** | ~50% | Gaze lock to camera. 100% = locked on user. 0% = natural wandering gaze | Animator → `liveportrait_driver.py` gaze_lock |
| **Flirtation** | 15% | Composite: smile intensity + brow play + head tilt frequency + voice breathiness | Animator + Voice combined |

**Left Column — Buttons**:
- **Backend Settings** (gold) → GPU profile selector, model swap, API keys, Redis/Celery status, memory dashboard
- **Design** row (8 waveform icons) → Voice profile presets. Each icon = different voice character (warm female, authoritative male, calm soothing, etc.). Click to swap TTS voice instantly

**Center Column — Pipeline Controls**:
- **Model to Model** → Live-swap any model mid-session without restart. Switch CosyVoice2 → StyleTTS2 or LivePortrait Path A → HunyuanVideo-Avatar Path B
- **New Pipeline** → Pipeline builder (React Flow node editor). Drag-and-drop model nodes to create custom inference chains. Five archetypes:
  1. **Low-Latency Streamer**: WebRTC → Whisper-Small → BitNet-3B → Kokoro → LivePortrait
  2. **Emotive Actor**: Sentiment-Analyzer → Emotion-LoRA-Router → StyleTTS2 → HunyuanAvatar
  3. **Knowledge Expert RAG**: Vector-DB → Context-Injection → Claude Sonnet → CosyVoice2 → LivePortrait
  4. **Zero-Shot Creator**: User-Image-Upload → IP-Adapter → FLUX → LivePortrait
  5. **Director's Cut**: Human-in-the-Loop → Manual-Pose-Control → LivePortrait
- **Connectivity** → Real-time status: WebRTC CONNECTED/DISCONNECTED, WebSocket fallback, GPU util, active models

**Right Column — Avatar + Context**:
- **EVE portrait** — live animated video feed during conversation
- **EDEN** pill button (top-right) → Avatar identity selector. Swap between different avatar models
- **Custom Instructions & Context Ref** overlay → System prompt editor overlaying EVE. Markdown-supported persona instructions. Shows active context document
- **Apply to EVE's Memory** → Commits instructions to Brain's persona manager + persists key facts to long-term memory
- **Compliance Matter** badge → Visual indicator that persona is compliance-reviewed (medical/financial)

**Bottom Row — Action Buttons**:
- **Build Voice Agent** → Voice agent creation wizard: template → persona → voice clone → appearance → deploy
- **Hair & Wardrobe** → Appearance editor. Changes hair, clothing, accessories, background via FLUX inpainting with IP-Adapter identity lock (face preserved, outfit changed)
- **THE VOICE** (large gold bar) → Full voice config: cloning upload, emotion sliders, speed, language, preview
- **Initiate Conversation** (large gold bar) → Primary CTA. Boots pipeline, starts idle loop, activates ASR, enters conversation mode

### Knowledge Injection Modal (Admin Panel 2)

This modal is the **intelligence layer**. It feeds EVE domain knowledge so she can discuss specific content with authority. This is the Naoma-killer: instead of a cartoon avatar reading a sales script, EVE is a photorealistic human who has consumed your product demos, audiobooks, and research.

**Input Fields**:

1. **YouTube URL Input** (with Paste button)
   - Paste any YouTube URL. System extracts full transcript via `yt-dlp` + Whisper, key topics with timestamps, visual descriptions of product UI via frame sampling + vision model
   - Injected into Brain's knowledge base as structured context
   - EVE can: "Let me walk you through what was shown at the 3:42 mark of that demo..."
   - **Use case**: Feed product demo video → EVE becomes 24/7 sales agent who discusses every feature like a human colleague who watched the video

2. **Audiobook / Media URL Input** (with Upload button)
   - Upload MP3/WAV/M4A or paste media URLs
   - Full transcription via Whisper → semantic chunking → vector store embedding
   - EVE discusses themes, references passages, answers questions about content
   - **Use case**: Feed medical textbook audio → EVE tutors students on any concept from the book

3. **Research / Prompt URL**
   - Paste URL to arXiv paper, PDF, web article
   - Fetches content, extracts text, chunks and embeds in RAG store
   - EVE discusses findings, compares methodologies, explains concepts
   - **Use case**: Feed company whitepaper → EVE presents your research as subject-matter expert

4. **Natural Language Prompt for Prototyping** (large textarea)
   - Free-form meta-instructions for building new agent behaviors
   - Example: "Create a conversational agent with VASA-1 level realness inspired by the Teller and Soul papers..."
   - Tells the Brain engine how to configure itself. Supports model/paper references
   - **Send Prompt** button fires instruction to Conductor

5. **Analyze Media Sources** (gold bar button)
   - Batch processes all ingested media: transcription → chunking → embedding → knowledge graph construction
   - Shows progress and summary of extracted knowledge
   - Once complete, EVE's Brain has the full knowledge base loaded and ready

### What Makes EDEN OS Different from Naoma

| Capability | Naoma | EDEN OS |
|-----------|-------|---------|
| Avatar realism | Cartoon/basic | Photorealistic 4D human (Eden Protocol) |
| Lip-sync | Basic mouth movement | Phoneme-accurate LivePortrait at 78fps |
| Knowledge sources | Sales script + KB | YouTube + audiobooks + research papers + live URLs |
| Interruption handling | Limited | Full KV-Recache protocol (<100ms) |
| Real-time tuning | None | Live behavioral sliders (expressiveness, eye contact, flirtation) |
| Voice | Standard TTS | CosyVoice2 zero-shot cloning + emotion routing |
| Product demos | Script playback | Contextual video discussion with timestamp references |
| Deployment | Cloud only | Self-hosted on RTX 3090+, data never leaves your machine |

---

## AGENT TEAM SPECIFICATION

This project is built by a team of **7 specialized Claude Code agents** working in parallel. Each agent owns a vertical slice of the system. Agents communicate through shared file interfaces and a central orchestration manifest.

---

### AGENT 1: GENESIS (Portrait-to-4D Engine)

**Role**: Owns the image generation and 4D avatar creation pipeline.
**Objective**: Convert any 2D image into a temporally-consistent 4D avatar mesh/latent that can be animated in real-time.

**Model Stack (ordered by priority)**:
| Model | Purpose | HF Repo | VRAM | Latency |
|-------|---------|---------|------|---------|
| FLUX 1.0 Pro | Portrait generation/enhancement | `black-forest-labs/FLUX.1-pro` | 22GB | 4-6s |
| FLUX.1-schnell | Fast preview / real-time feedback | `black-forest-labs/FLUX.1-schnell` | 14GB | 1-2s |
| IP-Adapter FaceID | Identity preservation from upload | `h94/IP-Adapter-FaceID` | 2GB | <1s |
| RealESRGAN x4 | Background upscale (async) | `ai-forever/Real-ESRGAN` | 1GB | <1s |

**Tasks**:
1. Build `genesis/portrait_engine.py` — accepts uploaded image, runs face detection (MediaPipe or InsightFace), crops and aligns face, generates enhanced portrait via FLUX with IP-Adapter for identity lock
2. Build `genesis/eden_protocol_validator.py` — implements the 0.3 deviation rule. Extracts micro-features (pores, freckles, beauty marks) from reference and generated images. Rejects and regenerates if deviation exceeds threshold
3. Build `genesis/latent_encoder.py` — encodes the portrait into the latent space compatible with the animation engine (MuseTalk/HunyuanVideo-Avatar latent format)
4. Build `genesis/preload_cache.py` — pre-computes the avatar's idle animations (blinks, micro-movements, breathing) so the avatar is alive on page load with ZERO wait time

**Critical Constraint**: The portrait must be generated/processed and cached BEFORE the user initiates conversation. The "ready on load" requirement means Genesis runs during the setup phase, not during chat.

**File Output**: `genesis/` directory with all modules. Exports a `GenesisEngine` class with methods: `process_upload()`, `generate_portrait()`, `validate_eden_protocol()`, `encode_to_latent()`, `precompute_idle_cache()`

---

### AGENT 2: VOICE (TTS + Voice Cloning + ASR)

**Role**: Owns all audio — speech recognition, voice synthesis, voice cloning, emotion injection.
**Objective**: Deliver emotionally-aware, sub-200ms TTS with optional voice cloning from 10-second reference audio.

**Model Stack**:
| Model | Purpose | HF Repo | VRAM | Latency |
|-------|---------|---------|------|---------|
| Whisper Large v3 Turbo | Real-time ASR | `openai/whisper-large-v3-turbo` | 3GB | <500ms |
| CosyVoice 2 | Primary TTS + zero-shot cloning | `FunAudioLLM/CosyVoice2-0.5B` | 2GB | <300ms |
| Kokoro v1.0 | Fallback baseline TTS | `hexgrad/Kokoro-82M` | <1GB | <200ms |
| StyleTTS2 | Emotional voice cloning (premium) | `yl4579/StyleTTS2` | 4GB | 2-4s |
| Silero VAD | Voice Activity Detection | `snakers5/silero-vad` | <1GB | <10ms |

**Tasks**:
1. Build `voice/asr_engine.py` — real-time speech-to-text using Whisper with Silero VAD for endpoint detection. Must support streaming (partial transcripts while user is still speaking). Implements the LISTENING state audio pipeline
2. Build `voice/tts_engine.py` — text-to-speech with CosyVoice 2 as primary (supports zero-shot cloning from 3-10s reference). Kokoro as fallback. Must support streaming output (begin generating audio from first LLM tokens, don't wait for complete response)
3. Build `voice/voice_cloner.py` — async voice cloning pipeline. Accepts reference audio, extracts voice embedding, stores for future TTS calls. Emotion dict: `{joy, sadness, confidence, urgency, warmth}` each 0.0-1.0
4. Build `voice/emotion_router.py` — analyzes LLM response text sentiment and automatically adjusts TTS emotion parameters. A medical agent should sound warm and reassuring
5. Build `voice/interruption_handler.py` — detects when the user begins speaking while the avatar is still talking. Immediately signals the orchestrator to halt TTS generation, flush the audio buffer, and transition to LISTENING state

**Critical Constraint**: TTS must begin streaming audio from the FIRST LLM token. Do NOT wait for the full LLM response. This is what makes the avatar feel "alive" — it starts speaking as it thinks, just like a human.

**File Output**: `voice/` directory. Exports a `VoiceEngine` class with methods: `start_listening()`, `stop_listening()`, `synthesize_stream()`, `clone_voice()`, `detect_interruption()`

---

### AGENT 3: ANIMATOR (Lip-Sync + 4D Motion Engine)

**Role**: Owns all facial animation — lip-sync, head motion, micro-expressions, idle behavior, and the critical state transitions (LISTENING ↔ THINKING ↔ SPEAKING).
**Objective**: Generate 60fps photorealistic facial animation driven by audio, synchronized with TTS output, with zero uncanny valley.

**Model Stack (Dual-Path Architecture)**:

**Path A — Implicit Keypoint Path (PRIMARY, for real-time)**:
| Model | Purpose | HF Repo | Speed | VRAM |
|-------|---------|---------|-------|------|
| LivePortrait | Implicit keypoint extraction + stitching + retargeting | `KwaiVGI/LivePortrait` | 12.8ms/frame (78fps on RTX 4090) | 4GB |
| LivePortrait Retargeting MLP | Eyes + lips fine control via scalar inputs | Included in LivePortrait | <1ms | Negligible |

LivePortrait is the PRIMARY animation engine because:
- It uses implicit keypoints (compact blendshapes) rather than heavy diffusion, achieving 12.8ms per frame
- The stitching module seamlessly pastes the animated face back into the original image — no shoulder glitches, no border artifacts
- Eyes retargeting and lip retargeting modules accept scalar inputs, giving us precise programmatic control over gaze direction and mouth shape
- Trained on 69 million high-quality frames with mixed image-video strategy — best-in-class generalization across ethnicities, art styles, and lighting
- The entire pipeline (appearance extractor → motion extractor → warping → decoder → stitching) runs in under 13ms

**Path B — Diffusion Path (PREMIUM, for cinematic quality)**:
| Model | Purpose | HF Repo | Speed | VRAM |
|-------|---------|---------|-------|------|
| HunyuanVideo-Avatar | MM-DiT audio-driven animation with emotion control | `tencent/HunyuanVideo-Avatar` | ~2s per clip | 16GB |
| MuseTalk v2 | Latent-space lip-sync inpainting | `TMElyralab/MuseTalk` | 30fps on V100 | 4GB |
| Hallo 3 | Diffusion transformer portrait animation | Community | ~3s per clip | 12GB |

**Path Selection Logic**:
- Real-time conversation → Path A (LivePortrait) — always
- Pre-rendered cinematic content → Path B (HunyuanVideo-Avatar)
- Fallback if LivePortrait fails → MuseTalk v2

**Tasks**:
1. Build `animator/liveportrait_driver.py` — wraps LivePortrait's inference pipeline. Accepts audio features (from Voice engine) and converts them to implicit keypoint deltas for lip retargeting. Maps phoneme sequences to mouth shapes via the lip retargeting MLP. Handles eye blink injection, gaze direction from user webcam (if available), and natural head sway
2. Build `animator/idle_generator.py` — generates the LISTENING state idle loop. Uses LivePortrait's retargeting modules to produce natural blinks (every 3-7 seconds, randomized), micro head movements (+/-2 degrees rotation), subtle breathing motion (chest/shoulder rise), and occasional eyebrow micro-raises. This loop runs CONTINUOUSLY when the avatar is not speaking
3. Build `animator/state_machine.py` — manages transitions between the three states of presence:
   - LISTENING → THINKING: triggered by ASR endpoint detection. Avatar does a subtle inhale, slight brow raise
   - THINKING → SPEAKING: triggered by first TTS audio chunk. Avatar opens mouth, begins lip-sync
   - SPEAKING → LISTENING: triggered by TTS completion or user interruption. Avatar closes mouth, returns to idle loop
   - SPEAKING → LISTENING (INTERRUPT): triggered by `interruption_handler`. Immediate halt, smooth transition back to idle within 100ms using KV-recache technique adapted from LONGLIVE
4. Build `animator/audio_to_keypoints.py` — the critical bridge between Voice and Animator. Converts audio waveform features (mel spectrogram, pitch, energy) into LivePortrait-compatible implicit keypoint deltas. This replaces the need for a "driving video" — audio becomes the driver
5. Build `animator/eden_temporal_anchor.py` — implements the temporal consistency system adapted from LONGLIVE's frame sink concept. Always maintains the first frame of each conversation turn as a "global anchor" so the avatar never drifts from its identity over long conversations. Prevents the "latent collapse" phenomenon where AI faces slowly lose their identity

**Critical Constraint**: The animator must NEVER produce a frozen frame. Even during model loading or state transitions, the idle loop must continue. The avatar is always alive.

**File Output**: `animator/` directory. Exports an `AnimatorEngine` class with methods: `start_idle_loop()`, `drive_from_audio()`, `transition_state()`, `get_current_frame()`, `apply_eden_anchor()`

---

### AGENT 4: BRAIN (LLM Reasoning + Context Engine)

**Role**: Owns the conversational intelligence — LLM integration, system prompts, memory, context management, and persona behavior.
**Objective**: Deliver context-aware, persona-consistent responses with sub-200ms first-token latency via streaming.

**Model Stack (Tiered)**:
| Model | Purpose | Provider | Latency | Cost |
|-------|---------|----------|---------|------|
| Claude Sonnet 4 | Primary reasoning (cloud) | Anthropic API | <150ms first token | $0.003/1K tokens |
| Qwen 3 8B (GGUF Q4) | Local fallback / offline mode | `Qwen/Qwen3-8B-GGUF` via llama.cpp | <300ms first token | $0 |
| BitNet b1.58 3B | Ultra-efficient edge mode | `microsoft/BitNet` via llama.cpp | <200ms first token | $0 |

**Tiered Selection**:
- Internet available + API key configured → Claude Sonnet 4 (best quality)
- Offline or API failure → Qwen 3 8B via llama.cpp (good quality, runs on CPU+GPU)
- Edge deployment / mobile / low-VRAM → BitNet 3B (acceptable quality, runs on CPU only, frees GPU for animation)

**Tasks**:
1. Build `brain/reasoning_engine.py` — LLM interface with streaming response. Must yield tokens as they arrive (not wait for complete response). Supports both Anthropic API (cloud) and llama.cpp (local). Handles system prompt injection, conversation history, and persona context
2. Build `brain/persona_manager.py` — loads agent persona from YAML template files. Each persona defines: name, role, tone, knowledge domain, emotional baseline, conversation boundaries. The persona shapes every response
3. Build `brain/memory_manager.py` — maintains conversation history within session. Implements sliding window context (last 20 turns). Extracts key facts mentioned by user for context persistence. Future: vector DB integration for long-term memory
4. Build `brain/streaming_bridge.py` — the critical integration point. As LLM tokens stream in, this module:
   - Buffers tokens until a natural speech boundary (sentence end, comma pause, etc.)
   - Sends each buffer to Voice engine for TTS generation
   - Voice engine sends audio chunks to Animator engine for lip-sync
   - Result: the avatar begins speaking within 500ms of the user finishing their question
5. Build `brain/template_loader.py` — loads and validates agent templates (YAML). Templates define the full agent configuration: persona, voice profile, visual appearance preferences, knowledge base references

**Template Schema**:
```yaml
# templates/medical_office.yaml
agent:
  name: "Dr. Rivera's Assistant"
  role: "Medical office receptionist"
  persona:
    tone: warm
    pace: moderate
    formality: professional
    emotional_baseline: {joy: 0.6, confidence: 0.8, warmth: 0.9}
  system_prompt: |
    You are a warm, professional medical office assistant for Dr. Rivera's
    family practice. You help patients schedule appointments, answer general
    questions about office hours and services, and collect basic intake
    information. You are HIPAA-aware and never discuss other patients.
    You speak clearly and reassuringly.
  voice:
    engine: cosyvoice2
    reference_audio: null  # uses default warm female voice
    speed: 0.95
    emotion_override: {warmth: 0.9, confidence: 0.7}
  appearance:
    portrait_prompt: "Professional woman, warm smile, medical office background"
    style: photorealistic
    eden_protocol: strict
  knowledge_base:
    - office_hours.md
    - services.md
    - insurance_accepted.md
```

**File Output**: `brain/` directory. Exports a `BrainEngine` class with methods: `reason_stream()`, `load_persona()`, `get_context()`, `process_user_input()`

---

### AGENT 5: CONDUCTOR (Pipeline Orchestrator + State Manager)

**Role**: Owns the end-to-end orchestration — connects all engines, manages data flow, handles errors, enforces latency budgets, and serves as the single entry point for the system.
**Objective**: Orchestrate the full pipeline from user input to avatar video output in under 5 seconds total, with the avatar appearing alive and responsive at all times.

**Tasks**:
1. Build `conductor/orchestrator.py` — the master controller. Implements the full pipeline.
2. Build `conductor/latency_enforcer.py` — monitors each pipeline stage and enforces latency budgets.
3. Build `conductor/error_recovery.py` — handles failures gracefully.
4. Build `conductor/session_manager.py` — manages the lifecycle of a conversation session.
5. Build `conductor/metrics_collector.py` — collects real-time performance metrics.

**File Output**: `conductor/` directory. Exports a `Conductor` class as the single entry point: `Conductor(config).create_session().start_conversation()`

---

### AGENT 6: GATEWAY (WebRTC Server + API Layer)

**Role**: Owns the network layer — WebRTC signaling, video/audio streaming, REST API for session management, and the frontend connection.
**Objective**: Stream the avatar video to the user's browser at 60fps with sub-500ms latency, handle audio input capture, and provide a clean API for session lifecycle.

**Tasks**:
1. Build `gateway/api_server.py` — FastAPI application with all endpoints from API Contract
2. Build `gateway/webrtc_handler.py` — WebRTC signaling and media transport
3. Build `gateway/audio_capture.py` — processes incoming WebRTC audio
4. Build `gateway/video_encoder.py` — encodes animator output frames to streamable video
5. Build `gateway/websocket_handler.py` — WebSocket fallback streaming

**File Output**: `gateway/` directory. Exports a `GatewayServer` class with method: `start(host, port)` that boots the entire API + WebRTC server

---

### AGENT 7: SCHOLAR (Knowledge Engine + Media Ingestion)

**Role**: Owns all knowledge ingestion — YouTube transcription, audiobook processing, research paper parsing, URL scraping, RAG vector store, and the knowledge graph that makes EVE an expert on any topic you feed her.
**Objective**: Transform any media source (video, audio, text, URL) into structured knowledge that the Brain engine can retrieve during conversation, with citation-level accuracy.

**Tasks**:
1. Build `scholar/youtube_ingestor.py` — the YouTube knowledge pipeline
2. Build `scholar/audiobook_ingestor.py` — audiobook and media processing
3. Build `scholar/url_ingestor.py` — web and research paper ingestion
4. Build `scholar/knowledge_graph.py` — connects ingested knowledge
5. Build `scholar/rag_retriever.py` — the retrieval interface for the Brain
6. Build `scholar/media_analyzer.py` — the "Analyze Media Sources" button handler

**File Output**: `scholar/` directory. Exports a `ScholarEngine` class with methods: `ingest_youtube()`, `ingest_audiobook()`, `ingest_url()`, `analyze_all()`, `retrieve()`, `get_knowledge_summary()`

---

## MODEL PRIORITY MATRIX

```
| HARDWARE TIER     | ANIMATION       | TTS           | LLM      |
|-------------------|-----------------|---------------|----------|
| H100 (80GB)       | HunyuanAvatar   | CosyVoice2    | Claude   |
| RTX 4090 (24GB)   | LivePortrait    | CosyVoice2    | Claude   |
| RTX 3090 (24GB)   | LivePortrait    | Kokoro        | Qwen3 8B |
| L4 (24GB)         | LivePortrait    | Kokoro        | Claude   |
| CPU Only          | LivePortrait    | Kokoro        | BitNet 3B|
```

---

## ENVIRONMENT AND DEPENDENCIES

### requirements.txt
```
# Core
fastapi==0.115.0
uvicorn==0.30.0
websockets==12.0
pydantic==2.9.0

# ML / Inference
torch==2.4.0
torchaudio==2.4.0
torchvision==0.19.0
transformers==4.45.0
diffusers==0.31.0
accelerate==0.34.0
safetensors==0.4.5
huggingface-hub==0.25.0

# LivePortrait dependencies
insightface==0.7.3
onnxruntime-gpu==1.19.0
mediapipe==0.10.14

# Voice
openai-whisper==20231117
silero-vad==5.1

# WebRTC
aiortc==1.9.0

# Image/Video processing
opencv-python-headless==4.10.0
Pillow==10.4.0
numpy==1.26.4
scipy==1.14.0
scikit-image==0.24.0

# Utilities
pyyaml==6.0.2
anthropic==0.34.0
loguru==0.7.2

# Scholar / Knowledge Engine (Agent 7)
yt-dlp==2024.10.22
chromadb==0.5.5
sentence-transformers==3.1.0
trafilatura==1.12.0
pymupdf==1.24.10
```

---

**OWN THE SCIENCE.**
**EDEN OS v1.0 — Phase One**
**Built by Amanda + 7 Claude Code Agents**

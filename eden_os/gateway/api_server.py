"""
EDEN OS -- Gateway API Server (Agent 6)
FastAPI application with all REST + WebSocket endpoints.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from eden_os.gateway.websocket_handler import WebSocketHandler

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    portrait_image: Optional[str] = Field(None, description="Base64-encoded portrait image")
    template: str = Field("default", description="Agent template name")


class CreateSessionResponse(BaseModel):
    session_id: str
    ws_url: str
    status: str = "ready"


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    metrics: dict = Field(default_factory=dict)


class SettingsUpdateRequest(BaseModel):
    expressiveness: Optional[float] = None
    eye_contact: Optional[float] = None
    voice_tone: Optional[float] = None
    consistency: Optional[float] = None
    latency: Optional[float] = None
    flirtation: Optional[float] = None


class SettingsUpdateResponse(BaseModel):
    applied: bool = True


class PipelineSwapRequest(BaseModel):
    tts_engine: Optional[str] = None
    animation_engine: Optional[str] = None


class PipelineSwapResponse(BaseModel):
    swapped: bool = True
    reload_time_ms: float = 0.0


class KnowledgeIngestRequest(BaseModel):
    type: str = Field(..., description="youtube | audiobook | url")
    url: Optional[str] = None
    file: Optional[str] = None  # base64 for audiobook


class KnowledgeIngestResponse(BaseModel):
    job_id: str
    status: str = "processing"
    chunks_estimated: int = 0


class TemplateInfo(BaseModel):
    name: str
    description: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu: dict = Field(default_factory=dict)
    active_sessions: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Application state (held on the app instance via app.state)
# ---------------------------------------------------------------------------


class _AppState:
    """Mutable state attached to the FastAPI app."""

    def __init__(self, hardware_profile: str, models_cache: str):
        self.hardware_profile = hardware_profile
        self.models_cache = models_cache
        self.boot_time: float = time.time()
        # session_id -> session dict
        self.sessions: dict[str, dict[str, Any]] = {}
        # Conductor is lazy-initialized the first time a session is created
        self._conductor: Any = None
        # Scholar is lazy-initialized on first knowledge ingest
        self._scholar: Any = None
        # WebSocket handler pool: session_id -> WebSocketHandler
        self.ws_handlers: dict[str, WebSocketHandler] = {}

    # ------------------------------------------------------------------
    @property
    def conductor(self) -> Any:
        """Lazy-load the Conductor so the gateway can start even if
        other engines are not yet installed."""
        if self._conductor is None:
            try:
                from eden_os.conductor import Conductor  # type: ignore[import-untyped]

                self._conductor = Conductor(
                    hardware_profile=self.hardware_profile,
                    models_cache=self.models_cache,
                )
                logger.info("Conductor lazy-initialized successfully.")
            except Exception as exc:
                logger.warning(
                    f"Conductor not available (stub mode): {exc}"
                )
                self._conductor = _StubConductor()
        return self._conductor

    @property
    def scholar(self) -> Any:
        if self._scholar is None:
            try:
                from eden_os.scholar import ScholarEngine

                self._scholar = ScholarEngine()
                logger.info("ScholarEngine lazy-initialized successfully.")
            except Exception as exc:
                logger.warning(
                    f"ScholarEngine not available (stub mode): {exc}"
                )
                self._scholar = None
        return self._scholar


class _StubConductor:
    """Minimal stub so the API can respond even when the real Conductor
    is not installed yet."""

    async def create_session(self, config: Any) -> str:
        return str(uuid.uuid4())

    async def start_conversation(self, session_id: str) -> None:
        pass

    async def end_conversation(self, session_id: str) -> None:
        pass

    async def get_metrics(self, session_id: str) -> dict:
        return {}

    async def interrupt(self, session_id: str) -> None:
        pass

    async def update_settings(self, session_id: str, settings: dict) -> None:
        pass

    async def swap_pipeline(self, session_id: str, swap: dict) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


def _list_templates() -> list[TemplateInfo]:
    """Scan the templates/ directory for YAML files."""
    templates: list[TemplateInfo] = []
    if _TEMPLATES_DIR.is_dir():
        for p in sorted(_TEMPLATES_DIR.glob("*.yaml")):
            templates.append(
                TemplateInfo(name=p.stem, description=f"Template: {p.stem}")
            )
    if not templates:
        templates.append(TemplateInfo(name="default", description="Default EVE template"))
    return templates


# ---------------------------------------------------------------------------
# GPU info helper
# ---------------------------------------------------------------------------


def _gpu_info() -> dict:
    """Best-effort GPU info via torch.cuda."""
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return {
                "name": torch.cuda.get_device_name(idx),
                "memory_used_mb": round(
                    torch.cuda.memory_allocated(idx) / 1024 / 1024, 1
                ),
                "memory_total_mb": round(
                    torch.cuda.get_device_properties(idx).total_mem / 1024 / 1024, 1
                ),
            }
    except Exception:
        pass
    return {"name": "none", "memory_used_mb": 0, "memory_total_mb": 0}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    hardware_profile: str = "auto",
    models_cache: str = "models_cache",
) -> FastAPI:
    """Build and return the Gateway FastAPI application.

    Parameters
    ----------
    host:
        Bind address (informational — used when constructing ws_url).
    port:
        Bind port (informational — used when constructing ws_url).
    hardware_profile:
        Hardware tier string forwarded to the Conductor.
    models_cache:
        Path to the downloaded model weights directory.
    """

    app = FastAPI(
        title="EDEN OS Gateway",
        version="1.0.0",
        description="4D Bi-Directional Conversational Avatar API",
    )

    # -- CORS (allow all for dev) -----------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Shared state -----------------------------------------------------
    state = _AppState(hardware_profile=hardware_profile, models_cache=models_cache)
    app.state.eden = state
    # Store host/port for ws_url construction
    app.state.host = host
    app.state.port = port

    # -- Static files (EDEN Studio frontend) ------------------------------
    static_dir = Path(__file__).resolve().parents[2] / "static"
    _static_dir = static_dir  # capture for closure

    # ======================================================================
    # REST Endpoints
    # ======================================================================

    @app.post("/api/v1/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        ws_url = f"ws://{app.state.host}:{app.state.port}/api/v1/sessions/{session_id}/stream"

        state.sessions[session_id] = {
            "id": session_id,
            "template": req.template,
            "state": "ready",
            "settings": {
                "expressiveness": 0.6,
                "eye_contact": 0.5,
                "voice_tone": 0.85,
                "consistency": 0.7,
                "latency": 1.0,
                "flirtation": 0.15,
            },
            "created_at": time.time(),
        }

        # Delegate to the Conductor (lazy-loaded)
        try:
            from eden_os.shared.types import SessionConfig
            import numpy as np
            import base64

            # Decode the portrait image from base64 to numpy (best-effort)
            portrait_array = None
            if req.portrait_image:
                try:
                    raw = base64.b64decode(req.portrait_image)
                    portrait_array = np.frombuffer(raw, dtype=np.uint8)
                except Exception:
                    portrait_array = None

            config = SessionConfig(
                session_id=session_id,
                portrait_image=portrait_array,
                template_name=req.template,
                hardware_profile=state.hardware_profile,
            )
            await state.conductor.create_session(config)
        except Exception as exc:
            logger.warning(f"Conductor session creation deferred: {exc}")

        logger.info(f"Session created: {session_id} (template={req.template})")
        return CreateSessionResponse(
            session_id=session_id, ws_url=ws_url, status="ready"
        )

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        """End and clean up a session."""
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Clean up WebSocket handler if active
        handler = state.ws_handlers.pop(session_id, None)
        if handler is not None:
            await handler.close()

        try:
            await state.conductor.end_conversation(session_id)
        except Exception as exc:
            logger.warning(f"Conductor cleanup warning: {exc}")

        del state.sessions[session_id]
        logger.info(f"Session deleted: {session_id}")
        return {"deleted": True, "session_id": session_id}

    @app.get(
        "/api/v1/sessions/{session_id}/status",
        response_model=SessionStatusResponse,
    )
    async def session_status(session_id: str) -> SessionStatusResponse:
        """Return current session state and pipeline metrics."""
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = state.sessions[session_id]
        metrics: dict = {}
        try:
            raw_metrics = await state.conductor.get_metrics(session_id)
            if hasattr(raw_metrics, "__dict__"):
                metrics = {
                    k: v
                    for k, v in raw_metrics.__dict__.items()
                    if not k.startswith("_")
                }
            elif isinstance(raw_metrics, dict):
                metrics = raw_metrics
        except Exception:
            pass

        return SessionStatusResponse(
            session_id=session_id,
            state=session.get("state", "unknown"),
            metrics=metrics,
        )

    @app.post("/api/v1/sessions/{session_id}/interrupt")
    async def interrupt_session(session_id: str) -> dict:
        """Force-interrupt the avatar (stop speaking, return to LISTENING)."""
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        state.sessions[session_id]["state"] = "listening"
        try:
            await state.conductor.interrupt(session_id)
        except Exception as exc:
            logger.warning(f"Conductor interrupt warning: {exc}")

        # Notify WebSocket handler
        handler = state.ws_handlers.get(session_id)
        if handler is not None:
            await handler.handle_interrupt()

        logger.info(f"Session interrupted: {session_id}")
        return {"interrupted": True}

    @app.put(
        "/api/v1/sessions/{session_id}/settings",
        response_model=SettingsUpdateResponse,
    )
    async def update_settings(
        session_id: str, req: SettingsUpdateRequest
    ) -> SettingsUpdateResponse:
        """Update behavioral sliders in real-time."""
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        updates = req.model_dump(exclude_none=True)
        state.sessions[session_id]["settings"].update(updates)

        try:
            await state.conductor.update_settings(session_id, updates)
        except Exception as exc:
            logger.warning(f"Conductor settings update warning: {exc}")

        logger.info(f"Settings updated for {session_id}: {updates}")
        return SettingsUpdateResponse(applied=True)

    @app.put(
        "/api/v1/sessions/{session_id}/pipeline",
        response_model=PipelineSwapResponse,
    )
    async def swap_pipeline(
        session_id: str, req: PipelineSwapRequest
    ) -> PipelineSwapResponse:
        """Swap models mid-session without restart."""
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        swap_config = req.model_dump(exclude_none=True)
        reload_ms = 0.0
        try:
            reload_ms = await state.conductor.swap_pipeline(session_id, swap_config)
        except Exception as exc:
            logger.warning(f"Pipeline swap warning: {exc}")

        logger.info(f"Pipeline swapped for {session_id}: {swap_config}")
        return PipelineSwapResponse(swapped=True, reload_time_ms=reload_ms)

    @app.get("/api/v1/templates", response_model=list[TemplateInfo])
    async def list_templates() -> list[TemplateInfo]:
        """List available agent templates."""
        return _list_templates()

    @app.post(
        "/api/v1/knowledge/ingest", response_model=KnowledgeIngestResponse
    )
    async def ingest_knowledge(
        req: KnowledgeIngestRequest,
    ) -> KnowledgeIngestResponse:
        """Ingest YouTube / audiobook / URL into the knowledge base."""
        job_id = str(uuid.uuid4())
        source = req.url or req.file or ""

        scholar = state.scholar
        if scholar is not None and hasattr(scholar, "queue_job"):
            try:
                job_id = scholar.queue_job(req.type, source)
            except Exception as exc:
                logger.warning(f"Scholar queue_job failed: {exc}")

        logger.info(f"Knowledge ingest queued: type={req.type}, job_id={job_id}")
        return KnowledgeIngestResponse(
            job_id=job_id, status="processing", chunks_estimated=0
        )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """System health check."""
        return HealthResponse(
            status="ok",
            gpu=_gpu_info(),
            active_sessions=len(state.sessions),
            uptime_seconds=round(time.time() - state.boot_time, 1),
        )

    # ======================================================================
    # WebSocket endpoint
    # ======================================================================

    @app.websocket("/api/v1/sessions/{session_id}/stream")
    async def websocket_stream(websocket: WebSocket, session_id: str) -> None:
        """Bi-directional audio/video streaming over WebSocket."""
        if session_id not in state.sessions:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        logger.info(f"WebSocket connected: {session_id}")

        handler = WebSocketHandler(
            session_id=session_id,
            websocket=websocket,
            conductor=state.conductor,
        )
        state.ws_handlers[session_id] = handler

        try:
            await handler.run()
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as exc:
            logger.error(f"WebSocket error ({session_id}): {exc}")
        finally:
            state.ws_handlers.pop(session_id, None)
            logger.info(f"WebSocket handler removed: {session_id}")

    # -- Mount static files + root route (MUST be after all API routes) ------
    if _static_dir.is_dir():
        from fastapi.responses import FileResponse

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(str(_static_dir / "index.html"))

        app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")
        logger.info(f"Static files mounted from {_static_dir}")
    else:
        logger.warning(f"Static directory not found at {_static_dir} — skipping mount.")

    return app

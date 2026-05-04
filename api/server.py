"""
api/server.py
--------------
FastAPI server — Genesis User Interaction Layer.

Endpoints:
  POST /generate            text → image
  POST /img2img             image + text → image
  GET  /result/{request_id} poll for async result
  POST /feedback            submit rating / correction
  GET  /models              list available models
  GET  /health              liveness probe
  GET  /stats               generation + feedback stats
  POST /admin/expand        trigger active learning expansion (admin)

All generation is async:
  submit → 202 Accepted + request_id
  poll   → GET /result/{id} returns status + image URLs when ready

Images are saved to outputs/api_images/<date>/<request_id>/
and served as static files under /images/<path>.

Authentication: Bearer token (optional, disabled by default).
Enabling: set GENESIS_API_KEY env variable.

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

    # Or via script:
    python -m api.server --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── FastAPI imports — graceful error if not installed ─────────
try:
    from fastapi import (
        Depends, FastAPI, File, Form, Header, HTTPException,
        Request, UploadFile, BackgroundTasks,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, validator
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.warning("FastAPI not installed — api/server.py will not function")

if _FASTAPI_OK:

    from inference.engine.model_router import GenerationRequest
    from inference.engine.optimized_pipeline import OptimizedPipeline, GenerationResult
    from inference.engine.batch_scheduler import BatchScheduler, JobStatus
    from api.prompt_logger import PromptLogger
    from api.feedback_store import FeedbackStore, FeedbackEntry

    # ── Pydantic models ───────────────────────────────────────

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=1000)
        negative_prompt: str = Field("", max_length=500)
        width: int = Field(512, ge=256, le=1024)
        height: int = Field(512, ge=256, le=1024)
        quality_tier: str = Field("fast", regex="^(fast|balanced|high)$")
        steps: Optional[int] = Field(None, ge=1, le=100)
        guidance: Optional[float] = Field(None, ge=0.0, le=30.0)
        seed: Optional[int] = None
        n_images: int = Field(1, ge=1, le=4)
        model_hint: Optional[str] = None
        user_id: str = "anonymous"

    class Img2ImgRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=1000)
        negative_prompt: str = ""
        image_base64: str = Field(..., description="Base64-encoded input image")
        strength: float = Field(0.75, ge=0.0, le=1.0)
        width: int = Field(512, ge=256, le=1024)
        height: int = Field(512, ge=256, le=1024)
        quality_tier: str = "balanced"
        steps: Optional[int] = None
        guidance: Optional[float] = None
        seed: Optional[int] = None
        n_images: int = 1
        user_id: str = "anonymous"

    class FeedbackRequest(BaseModel):
        request_id: str
        feedback_type: str = Field(..., regex="^(rating|thumbs|category|correction|report)$")
        rating: Optional[int] = Field(None, ge=1, le=5)
        thumbs_up: Optional[bool] = None
        category: Optional[str] = None
        correction: Optional[str] = Field(None, max_length=1000)
        note: Optional[str] = Field(None, max_length=500)
        image_path: Optional[str] = None
        user_id: str = "anonymous"

    class GenerateResponse(BaseModel):
        request_id: str
        status: str                      # queued | running | done | failed
        image_urls: List[str] = []
        model_used: str = ""
        duration_s: float = 0.0
        fallback_used: bool = False
        error: Optional[str] = None


    class LLMGenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=4000)
        max_new_tokens: int = Field(128, ge=1, le=1024)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        top_p: float = Field(0.9, ge=0.0, le=1.0)


    class LLMGenerateResponse(BaseModel):
        status: str
        output_text: str = ""
        model_used: str = ""
        backend: str = ""
        duration_s: float = 0.0
        error: Optional[str] = None

    # ── App factory ───────────────────────────────────────────

    def create_app(
        pipeline: "OptimizedPipeline",
        scheduler: "BatchScheduler",
        prompt_logger: "PromptLogger",
        feedback_store: "FeedbackStore",
        text_manager: Optional[Any] = None,
        images_dir: str = "outputs/api_images",
        api_key: Optional[str] = None,
        cors_origins: List[str] = ["*"],
    ) -> "FastAPI":
        app = FastAPI(
            title="Genesis AI Platform",
            description="Local generative AI — text→image, image→image",
            version="0.4.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static image serving
        images_path = Path(images_dir)
        images_path.mkdir(parents=True, exist_ok=True)
        app.mount("/images", StaticFiles(directory=str(images_path)), name="images")

        # ── Auth dependency ────────────────────────────────────
        def verify_token(authorization: Optional[str] = Header(None)) -> None:
            if api_key is None:
                return  # auth disabled
            if authorization is None or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing Bearer token")
            if authorization.split(" ", 1)[1] != api_key:
                raise HTTPException(status_code=403, detail="Invalid API key")

        # ── Helpers ────────────────────────────────────────────
        def _save_images(
            result: GenerationResult, request_id: str
        ) -> List[str]:
            date_dir = datetime.now().strftime("%Y-%m-%d")
            out_dir  = images_path / date_dir / request_id
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, img in enumerate(result.images):
                p = out_dir / f"img_{i:02d}.png"
                img.save(str(p), "PNG")
                paths.append(str(p))
            return paths

        def _image_url(path: str) -> str:
            rel = Path(path).relative_to(images_path)
            return f"/images/{rel.as_posix()}"

        def _on_complete(result: GenerationResult, orig_request: GenerateRequest) -> None:
            """Callback fired by BatchScheduler when a job finishes."""
            paths = _save_images(result, result.request_id)
            prompt_logger.log_result(
                request_id=result.request_id,
                prompt=orig_request.prompt,
                model_key=result.model_key,
                duration_s=result.duration_s,
                image_paths=paths,
                success=result.success,
                user_id=orig_request.user_id,
                negative_prompt=orig_request.negative_prompt,
                steps=result.steps,
                guidance=result.guidance,
                width=result.width,
                height=result.height,
                seed=result.seed,
                quality_tier=orig_request.quality_tier,
                fallback_used=result.fallback_used,
                error=result.error,
            )

        # ── Endpoints ──────────────────────────────────────────

        @app.get("/health")
        def health():
            return {"status": "ok", "timestamp": time.time()}

        @app.get("/models")
        def list_models(_=Depends(verify_token)):
            return {"models": pipeline.router.list_available()}

        @app.get("/stats")
        def stats(_=Depends(verify_token)):
            return {
                "generation": prompt_logger.stats(),
                "feedback":   feedback_store.stats(),
                "scheduler":  {
                    "submitted":  scheduler.stats().total_submitted,
                    "completed":  scheduler.stats().total_completed,
                    "queue_depth": scheduler.stats().queue_depth,
                },
                "pipeline": pipeline.status(),
            }


        @app.get("/llm/status")
        def llm_status(_=Depends(verify_token)):
            if text_manager is None:
                return {"enabled": False, "status": "disabled"}
            return {"enabled": True, **text_manager.status()}

        @app.get("/llm/models")
        def llm_models(_=Depends(verify_token)):
            return {
                "available_backends": ["llama_cpp", "airllm"],
                "configured_backend": getattr(text_manager, "backend", None),
            }

        @app.post("/llm/generate", response_model=LLMGenerateResponse)
        def llm_generate(req: LLMGenerateRequest, _=Depends(verify_token)):
            if text_manager is None:
                raise HTTPException(status_code=503, detail="LLM module disabled")
            try:
                result = text_manager.generate(
                    prompt=req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                return LLMGenerateResponse(**result)
            except Exception as e:
                return LLMGenerateResponse(status="failed", error=str(e))

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(
            req: GenerateRequest,
            _=Depends(verify_token),
        ):
            request_id = str(uuid.uuid4())[:12]
            gen_req = GenerationRequest(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width, height=req.height,
                quality_tier=req.quality_tier,
                steps=req.steps, guidance=req.guidance,
                seed=req.seed, n_images=req.n_images,
                request_id=request_id,
                user_id=req.user_id,
                model_hint=req.model_hint,
            )

            def callback(result: GenerationResult) -> None:
                _on_complete(result, req)

            job_id = scheduler.submit(gen_req, callback=callback)
            return GenerateResponse(
                request_id=job_id,
                status="queued",
            )

        @app.post("/generate/sync", response_model=GenerateResponse)
        async def generate_sync(
            req: GenerateRequest,
            _=Depends(verify_token),
        ):
            """Synchronous generation — blocks until complete. Use for low-latency clients."""
            request_id = str(uuid.uuid4())[:12]
            gen_req = GenerationRequest(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width, height=req.height,
                quality_tier=req.quality_tier,
                steps=req.steps, guidance=req.guidance,
                seed=req.seed, n_images=req.n_images,
                request_id=request_id,
                user_id=req.user_id,
                model_hint=req.model_hint,
            )
            result = pipeline.generate(gen_req)
            paths  = _save_images(result, request_id)
            _on_complete(result, req)

            return GenerateResponse(
                request_id=request_id,
                status="done" if result.success else "failed",
                image_urls=[_image_url(p) for p in paths],
                model_used=result.model_key,
                duration_s=result.duration_s,
                fallback_used=result.fallback_used,
                error=result.error,
            )

        @app.post("/img2img", response_model=GenerateResponse)
        async def img2img(
            req: Img2ImgRequest,
            _=Depends(verify_token),
        ):
            # Decode base64 image
            try:
                from PIL import Image as PILImage
                img_bytes = base64.b64decode(req.image_base64)
                init_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid image: {e}")

            request_id = str(uuid.uuid4())[:12]
            gen_req = GenerationRequest(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width, height=req.height,
                quality_tier=req.quality_tier,
                steps=req.steps, guidance=req.guidance,
                seed=req.seed, n_images=req.n_images,
                request_id=request_id,
                user_id=req.user_id,
                init_image=init_image,
                strength=req.strength,
            )
            result = pipeline.generate(gen_req)
            paths  = _save_images(result, request_id)

            fake_req_obj = type("R", (), {
                "prompt": req.prompt, "negative_prompt": req.negative_prompt,
                "quality_tier": req.quality_tier, "user_id": req.user_id,
            })()
            _on_complete(result, fake_req_obj)

            return GenerateResponse(
                request_id=request_id,
                status="done" if result.success else "failed",
                image_urls=[_image_url(p) for p in paths],
                model_used=result.model_key,
                duration_s=result.duration_s,
                fallback_used=result.fallback_used,
                error=result.error,
            )

        @app.get("/result/{request_id}", response_model=GenerateResponse)
        def get_result(request_id: str, _=Depends(verify_token)):
            job = scheduler.get_job(request_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Request not found")

            if job.status == JobStatus.QUEUED:
                return GenerateResponse(request_id=request_id, status="queued")
            if job.status == JobStatus.RUNNING:
                return GenerateResponse(request_id=request_id, status="running")
            if job.status == JobStatus.CANCELLED:
                return GenerateResponse(request_id=request_id, status="cancelled")

            result = job.result
            if result is None:
                return GenerateResponse(request_id=request_id, status="failed",
                                        error="No result")

            # Read saved image paths from prompt log
            entry = prompt_logger.get(request_id)
            image_urls = [_image_url(p) for p in (entry.image_paths if entry else [])]

            return GenerateResponse(
                request_id=request_id,
                status="done" if result.success else "failed",
                image_urls=image_urls,
                model_used=result.model_key,
                duration_s=result.duration_s,
                fallback_used=result.fallback_used,
                error=result.error,
            )

        @app.post("/feedback")
        def submit_feedback(
            req: FeedbackRequest,
            _=Depends(verify_token),
        ):
            # Enrich with prompt from log
            log_entry = prompt_logger.get(req.request_id)
            entry = FeedbackEntry(
                request_id=req.request_id,
                user_id=req.user_id,
                feedback_type=req.feedback_type,
                rating=req.rating,
                thumbs_up=req.thumbs_up,
                category=req.category,
                prompt=log_entry.prompt if log_entry else None,
                correction=req.correction,
                note=req.note,
                image_path=req.image_path,
            )
            fid = feedback_store.submit(entry)
            return {"feedback_id": fid, "status": "stored"}

        @app.get("/feedback/summary")
        def feedback_summary(_=Depends(verify_token)):
            return feedback_store.to_active_learning_signal()

        @app.post("/admin/expand")
        async def trigger_expansion(
            background_tasks: BackgroundTasks,
            _=Depends(verify_token),
        ):
            """Trigger active learning dataset expansion in background."""
            weak = feedback_store.weak_categories()
            if not weak:
                return {"status": "no_weak_categories", "categories": []}

            def _run_expansion():
                try:
                    from core.config import load_config
                    from dataset.active_learning.dataset_expander import DatasetExpander
                    cfg = GenesisConfig.load("configs/base.yaml")
                    expander = DatasetExpander.from_config(cfg)
                    result = expander.expand_from_feedback(weak)
                    logger.info(f"Admin expansion: {result.summary()}")
                except Exception as e:
                    logger.error(f"Admin expansion failed: {e}", exc_info=True)

            background_tasks.add_task(_run_expansion)
            return {
                "status": "expansion_queued",
                "weak_categories": weak,
            }

        @app.get("/admin/failures")
        def failure_analysis(_=Depends(verify_token)):
            return {
                "failed_prompts": prompt_logger.failed_prompts(limit=20),
                "weak_categories": feedback_store.weak_categories(),
                "corrections": feedback_store.correction_prompts(limit=10),
            }

        return app


    # ── Standalone launch ──────────────────────────────────────

    def build_and_run(
        host: str = "0.0.0.0",
        port: int = 8000,
        config_path: str = "configs/base.yaml",
        reload: bool = False,
    ) -> None:
        import uvicorn
        from core.config import load_config
        from core.logger import setup_logging

        setup_logging()
        cfg = load_config(config_path)

        from inference.engine.model_router import ModelRouter
        from inference.engine.optimized_pipeline import OptimizedPipeline
        from inference.engine.batch_scheduler import BatchScheduler

        pipeline = OptimizedPipeline(
            cfg=cfg,
            device=cfg.get_nested("system.device", "cpu"),
        )
        scheduler = BatchScheduler(pipeline, strategy="grouped")
        scheduler.start()

        api_key = os.environ.get("GENESIS_API_KEY")
        if api_key:
            logger.info("API key authentication enabled")

        text_manager = None
        if cfg.get_nested("llm.enabled", False):
            from llm import TextModelManager
            text_manager = TextModelManager(cfg)

        app = create_app(
            pipeline=pipeline,
            scheduler=scheduler,
            prompt_logger=PromptLogger(
                cfg.get_nested("api.log_db", "outputs/api_logs/prompts.db")
            ),
            feedback_store=FeedbackStore(
                cfg.get_nested("api.feedback_db", "outputs/api_logs/feedback.db")
            ),
            images_dir=cfg.get_nested("api.images_dir", "outputs/api_images"),
            api_key=api_key,
            text_manager=text_manager,
        )

        logger.info(f"Starting Genesis API on http://{host}:{port}")
        logger.info(f"  Docs: http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    if not _FASTAPI_OK:
        print("Install FastAPI: pip install fastapi uvicorn[standard]")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="Genesis API Server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    build_and_run(
        host=args.host, port=args.port,
        config_path=args.config, reload=args.reload,
    )

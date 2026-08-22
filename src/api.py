"""
src/api.py

FastAPI Web Backend for Voice-Enabled Multilingual RAG Application.
Exposes REST API endpoints for text Q&A (/api/text), voice Q&A (/api/voice),
and system health check (/health). Mounts static web frontend.

--------------------------------------------------------------------------------
ENDPOINTS:
  GET  /health     -> Health check status
  POST /api/text   -> Text-based Q&A query
  POST /api/voice  -> Voice-based Q&A query (audio file upload)
  GET  /           -> Serves frontend web app
--------------------------------------------------------------------------------
"""

import os
import sys
import uuid
import time
import shutil
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.dirname(__file__))

from rag_pipeline import RAGPipeline
from voice_rag_pipeline import VoiceRAGPipeline
from speech_to_text import SarvamSpeechToText

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(message)s")
logger = logging.getLogger("VoiceRAG_API")

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize FastAPI App
app = FastAPI(
    title="Voice-Enabled Multilingual RAG API",
    version="1.0.0",
    description="Multilingual RAG pipeline with Sarvam Speech-to-Text, FAISS, and BM25"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for pipelines
rag_pipeline_instance: Optional[RAGPipeline] = None
voice_pipeline_instance: Optional[VoiceRAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline_instance
    if rag_pipeline_instance is None:
        rag_pipeline_instance = RAGPipeline()
    return rag_pipeline_instance


def get_voice_pipeline() -> VoiceRAGPipeline:
    global voice_pipeline_instance
    if voice_pipeline_instance is None:
        rag_pipe = get_rag_pipeline()
        stt_provider = SarvamSpeechToText()
        voice_pipeline_instance = VoiceRAGPipeline(stt_provider=stt_provider, rag_pipeline=rag_pipe)
    return voice_pipeline_instance


# Middleware for Request Tracing & Structured Logging
@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    t_start = time.time()
    response: Response = await call_next(request)
    elapsed_ms = round((time.time() - t_start) * 1000, 2)
    
    response.headers["X-Request-ID"] = request_id
    logger.info(f"{request.method} {request.url.path} -> HTTP {response.status_code} ({elapsed_ms} ms)", extra={"request_id": request_id})
    return response


# Pydantic Input Schema for /api/text
class TextQueryRequest(BaseModel):
    query: str
    fast_mode: Optional[bool] = False


# 1. Health Endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "voice-rag"}


# 1b. Config Diagnostic Endpoint -- reports only presence/length of each
# generator API key (never the value) plus the actually-resolved provider,
# to debug "is the deployed secret actually wired up" without guessing from
# indirect query-response labels. Safe to leave public: no secret material
# is ever returned, only booleans/lengths.
@app.get("/api/debug/config")
async def debug_config():
    pipe = get_rag_pipeline()
    return {
        "GENERATOR_PROVIDER_env": os.getenv("GENERATOR_PROVIDER", ""),
        "GEMINI_API_KEY_present": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "GEMINI_API_KEY_len": len(os.getenv("GEMINI_API_KEY", "").strip()),
        "ANTHROPIC_API_KEY_present": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "GROQ_API_KEY_present": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "GROQ_API_KEY_len": len(os.getenv("GROQ_API_KEY", "").strip()),
        "resolved_generator_provider_name": pipe.generator.provider_name,
        "resolved_generator_class": type(pipe.generator).__name__,
    }


# 1c. Live Gemini connectivity probe -- makes one raw, uncached call
# straight to the Gemini API from the running Fly machine (bypassing the
# retry/fallback chain) and returns the real HTTP status + error body, to
# tell apart a bad key from rate-limiting from an outbound-network/region
# problem, none of which are distinguishable from query-response labels
# alone (every failure mode collapses to the same Groq/local fallback).
@app.get("/api/debug/gemini-live")
async def debug_gemini_live():
    import requests
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")
    if not api_key:
        return {"ok": False, "reason": "GEMINI_API_KEY not set in this process"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    t0 = time.time()
    try:
        resp = requests.post(
            url, headers={"content-type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}]},
            params={"key": api_key}, timeout=15.0,
        )
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        return {
            "ok": resp.status_code == 200,
            "http_status": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "body": resp.text[:600],
        }
    except Exception as e:
        return {"ok": False, "exception": type(e).__name__, "message": str(e)}


# 2. Text Q&A Endpoint
@app.post("/api/text")
async def text_query(req: TextQueryRequest, request: Request):
    req_id = getattr(request.state, "request_id", "internal")
    query_text = req.query.strip() if req.query else ""
    
    logger.info(f"Received text query: '{query_text[:50]}...'", extra={"request_id": req_id})
    
    try:
        pipeline = get_rag_pipeline()
        result = pipeline.answer(query_text, fast_mode=bool(req.fast_mode))
        
        # Inject request_id into metadata
        result.setdefault("metadata", {})["request_id"] = req_id
        return result
    except Exception as e:
        logger.error(f"Unhandled exception in /api/text: {e}", exc_info=True, extra={"request_id": req_id})
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "query": query_text,
                "answer": "An internal server error occurred while processing your request.",
                "grounded": False,
                "guardrail_reason": "Internal server error",
                "retrieved_context": [],
                "latency": {"total_ms": 0.0},
                "metadata": {"request_id": req_id}
            }
        )


# 3. Voice Q&A Endpoint (Audio Upload)
@app.post("/api/voice")
async def voice_query(request: Request, file: UploadFile = File(...)):
    req_id = getattr(request.state, "request_id", "internal")
    logger.info(f"Received audio file upload: '{file.filename}' ({file.content_type})", extra={"request_id": req_id})

    # Prepare temporary uploads directory
    temp_dir = os.path.join("data", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    # Determine file extension safely
    filename = file.filename or "recording.webm"
    ext = os.path.splitext(filename)[1].lower() or ".webm"
    temp_file_path = os.path.join(temp_dir, f"audio_{req_id}{ext}")

    try:
        # Save uploaded audio file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process through Voice RAG Pipeline
        voice_pipe = get_voice_pipeline()
        result = voice_pipe.answer_audio(temp_file_path)

        # Add request_id metadata
        result.setdefault("rag_details", {}).setdefault("metadata", {})["request_id"] = req_id
        result["request_id"] = req_id
        return result

    except Exception as e:
        logger.error(f"Unhandled exception in /api/voice: {e}", exc_info=True, extra={"request_id": req_id})
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "transcript": "",
                "answer": "An error occurred while processing the audio recording.",
                "language": "unknown",
                "grounded": False,
                "guardrail_reason": f"Server error: {str(e)}",
                "latency": {"stt_ms": 0.0, "rag_ms": 0.0, "total_ms": 0.0},
                "error": str(e),
                "request_id": req_id
            }
        )
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# 4. Serve Frontend Static Files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}

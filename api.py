"""
REST API for Semantic Search Engine
====================================
Production-grade FastAPI application with health checks, structured logging,
request validation, CORS, and OpenAPI documentation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or
    python api.py
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import get_settings
from semantic_search import SemanticSearchEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("sse.api")


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


def _reject_blank_items(values: list[str], field_name: str) -> list[str]:
    """Reject whitespace-only strings before they reach the embedding model."""
    blank_positions = [str(i) for i, value in enumerate(values) if not value.strip()]
    if blank_positions:
        raise ValueError(
            f"{field_name} must not contain blank text at index(es): {', '.join(blank_positions)}"
        )
    return values


class DocumentsRequest(BaseModel):
    """Request body for adding documents."""

    documents: list[str] = Field(
        ..., min_length=1, max_length=10_000, description="List of text documents to index"
    )
    batch_size: int = Field(default=64, ge=1, le=1024, description="Encoding batch size")

    @field_validator("documents")
    @classmethod
    def documents_must_not_be_blank(cls, value: list[str]) -> list[str]:
        """Return a 422 validation error for blank documents instead of a 500."""
        return _reject_blank_items(value, "documents")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": [
                        "Machine learning is fascinating",
                        "Python is a great language",
                    ],
                    "batch_size": 64,
                }
            ]
        }
    }


class SearchRequest(BaseModel):
    """Request body for a single search query."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query text")
    top_k: int = Field(default=5, ge=1, description="Number of results to return")
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "artificial intelligence", "top_k": 3, "threshold": 0.3}]
        }
    }


class BatchSearchRequest(BaseModel):
    """Request body for batch search."""

    queries: list[str] = Field(..., min_length=1, description="List of search queries")
    top_k: int = Field(default=5, ge=1, description="Results per query")

    @field_validator("queries")
    @classmethod
    def queries_must_not_be_blank(cls, value: list[str]) -> list[str]:
        """Validate every batch query before any search work is attempted."""
        return _reject_blank_items(value, "queries")

    model_config = {
        "json_schema_extra": {
            "examples": [{"queries": ["AI models", "web development"], "top_k": 3}]
        }
    }


class SearchResult(BaseModel):
    """A single search result."""

    document: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response for a search query."""

    query: str
    results: list[SearchResult]
    total_documents: int
    elapsed_ms: float


class BatchSearchResponse(BaseModel):
    """Response for batch search."""

    results: list[SearchResponse]
    elapsed_ms: float


class ApiLandingResponse(BaseModel):
    """Human-friendly API landing page for first-time users."""

    name: str
    version: str
    description: str
    docs_url: str
    health_url: str
    quick_start: list[str]
    example_request: dict[str, object]


class IndexStats(BaseModel):
    """Statistics about the current index."""

    total_documents: int
    model_name: str
    embedding_dim: int
    faiss_enabled: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    documents_indexed: int


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

# Module-level engine reference (populated during lifespan)
engine: SemanticSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    global engine
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("Initializing search engine (model=%s) ...", settings.model_name)
    engine = SemanticSearchEngine(
        model_name=settings.model_name,
        use_faiss=settings.use_faiss,
        normalize_embeddings=settings.normalize_embeddings,
    )

    # Optionally load a pre-built index
    if settings.index_path:
        try:
            engine = SemanticSearchEngine.load(settings.index_path)
            logger.info("Loaded index from %s (%d docs)", settings.index_path, len(engine))
        except Exception as exc:
            logger.warning("Could not load index from %s: %s", settings.index_path, exc)

    logger.info("Search engine ready")
    yield

    # Shutdown: auto-save if configured
    if settings.auto_save_path and engine and len(engine) > 0:
        logger.info("Auto-saving index to %s ...", settings.auto_save_path)
        engine.save(settings.auto_save_path)

    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="Semantic Search Engine API",
    description=(
        "A production-ready REST API for semantic similarity search, "
        "powered by sentence-transformers and FAISS."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Attach Server-Timing header with total request duration."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["Server-Timing"] = f"total;dur={elapsed_ms:.1f}"
    response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# --- Exception handlers ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _get_engine() -> SemanticSearchEngine:
    """Return the engine or raise 503 if not ready."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized")
    return engine


# --- Health & Info ---


@app.get("/", response_model=ApiLandingResponse, tags=["Health"])
async def api_landing():
    """Return a discoverable landing response for browsers and API clients."""
    return ApiLandingResponse(
        name="Semantic Search Engine API",
        version=settings.app_version,
        description="Index documents once, then retrieve the closest matches by meaning.",
        docs_url="/docs",
        health_url="/health",
        quick_start=[
            "POST /documents with {'documents': ['first text', 'second text']}",
            "POST /search with {'query': 'what you are looking for', 'top_k': 5}",
            "Use GET /search?q=your+query for browser-friendly testing",
        ],
        example_request={
            "method": "POST",
            "path": "/search",
            "json": {"query": "artificial intelligence", "top_k": 3},
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.

    Returns the service status, model state, and document count.
    """
    eng = _get_engine()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=eng.model is not None,
        documents_indexed=len(eng),
    )


@app.get("/stats", response_model=IndexStats, tags=["Health"])
async def index_stats():
    """Return statistics about the current search index."""
    eng = _get_engine()
    return IndexStats(
        total_documents=len(eng),
        model_name=eng.model_name,
        embedding_dim=eng.embedding_dim,
        faiss_enabled=eng.use_faiss,
    )


# --- Document Management ---


@app.post("/documents", tags=["Documents"], status_code=201)
async def add_documents(body: DocumentsRequest):
    """
    Add documents to the search index.

    Documents are encoded into embeddings and added to the vector store.
    Duplicate detection is **not** performed -- the caller is responsible
    for de-duplication.
    """
    eng = _get_engine()
    count_before = len(eng)
    eng.add_documents(body.documents, batch_size=body.batch_size, show_progress=False)

    # Auto-save
    if settings.auto_save_path:
        eng.save(settings.auto_save_path)

    return {
        "message": f"Added {len(body.documents)} documents",
        "total_documents": len(eng),
        "new_documents": len(eng) - count_before,
    }


@app.get("/documents/count", tags=["Documents"])
async def document_count():
    """Return the number of indexed documents."""
    eng = _get_engine()
    return {"count": len(eng)}


@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """Remove all documents and embeddings from the index."""
    eng = _get_engine()
    eng.clear()
    return {"message": "Index cleared", "total_documents": 0}


# --- Search ---


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(body: SearchRequest):
    """
    Perform a semantic search over indexed documents.

    Returns the top-k most similar documents ranked by cosine similarity.
    """
    eng = _get_engine()

    if len(eng) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Add documents first.")

    top_k = min(body.top_k, settings.max_top_k)

    start = time.perf_counter()
    raw_results = eng.search(body.query, top_k=top_k, threshold=body.threshold)
    elapsed = (time.perf_counter() - start) * 1000

    results = [
        SearchResult(document=doc, score=round(score, 4), rank=i + 1)
        for i, (doc, score) in enumerate(raw_results)
    ]

    return SearchResponse(
        query=body.query,
        results=results,
        total_documents=len(eng),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/search/batch", response_model=BatchSearchResponse, tags=["Search"])
async def search_batch(body: BatchSearchRequest):
    """
    Batch search: run multiple queries in a single request.

    Useful for comparing multiple queries or building recommendation matrices.
    """
    eng = _get_engine()

    if len(eng) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Add documents first.")

    if len(body.queries) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(f"Batch size {len(body.queries)} exceeds maximum of {settings.max_batch_size}"),
        )

    top_k = min(body.top_k, settings.max_top_k)

    overall_start = time.perf_counter()
    responses: list[SearchResponse] = []

    for query in body.queries:
        start = time.perf_counter()
        raw_results = eng.search(query, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000

        results = [
            SearchResult(document=doc, score=round(score, 4), rank=i + 1)
            for i, (doc, score) in enumerate(raw_results)
        ]
        responses.append(
            SearchResponse(
                query=query,
                results=results,
                total_documents=len(eng),
                elapsed_ms=round(elapsed, 2),
            )
        )

    total_elapsed = (time.perf_counter() - overall_start) * 1000
    return BatchSearchResponse(results=responses, elapsed_ms=round(total_elapsed, 2))


# --- Convenience GET search ---


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, max_length=10_000, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
):
    """
    GET-based search endpoint for simple integrations and browser testing.

    Example: ``/search?q=machine+learning&top_k=3``
    """
    return await search(SearchRequest(query=q, top_k=top_k, threshold=threshold))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )

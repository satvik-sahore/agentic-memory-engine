import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import settings
from src.db.qdrant import qdrant_manager
from src.api.routes import router as memory_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown routines."""
    logger.info("Initializing Self-Learning AI Agent Memory Engine...")
    is_ready = qdrant_manager.ensure_collection()
    logger.info(f"Vector Database status: {'Ready' if qdrant_manager.is_healthy() else 'Unavailable'}")
    yield
    logger.info("Shutting down Memory Engine...")
    qdrant_manager.close()


app = FastAPI(
    title="Self-Learning AI Agent Memory Service",
    description=(
        "Production-grade long-term memory engine with two-phase extraction, "
        "conflict reconciliation (ADD, UPDATE, DELETE, NOOP), and Qdrant vector storage."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend agents / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(memory_router)


@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint."""
    return {
        "name": "Self-Learning AI Agent Memory Service",
        "version": "0.1.0",
        "docs_url": "/docs",
        "status": "online",
        "provider": settings.provider,
        "extraction_model": settings.extraction_model,
        "embedding_model": settings.embedding_model,
    }


@app.get("/healthz", tags=["System"])
def health_check():
    """Health check for service and vector database."""
    db_healthy = qdrant_manager.is_healthy()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "qdrant_connected": db_healthy,
        "collection": settings.qdrant_collection_name,
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

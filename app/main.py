"""FastAPI application instance and router configuration."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.routes import analysis, payment, stats, admin, feedback
from .core.logging import setup_logging

# Setup logging
setup_logging(settings.log_dir)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Chat Psychology Analyzer API",
    description="API for analyzing chat conversations with psychological insights",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(analysis.router, tags=["analysis"])
app.include_router(stats.router, tags=["stats"])
app.include_router(feedback.router, tags=["feedback"])
if settings.payment_enabled:
    app.include_router(payment.router, prefix="/payment", tags=["payment"])
    
app.include_router(admin.router, prefix="/admin", tags=["admin"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Chat Psychology Analyzer API",
        "payment_enabled": settings.payment_enabled,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


logger.info("FastAPI application initialized")
logger.info(f"Payment mode: {'enabled' if settings.payment_enabled else 'disabled'}")

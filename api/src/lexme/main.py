"""FastAPI application entrypoint: builds the app, wires CORS and routers."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexme.api.ask import router as ask_router
from lexme.api.contract import router as contract_router
from lexme.api.corpus import router as corpus_router
from lexme.api.feedback import router as feedback_router
from lexme.api.health import router as health_router
from lexme.config import get_settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()
    app = FastAPI(title="Lexme API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(ask_router)
    app.include_router(contract_router)
    app.include_router(corpus_router)
    app.include_router(feedback_router)
    return app


app = create_app()

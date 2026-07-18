"""FastAPI application entrypoint: builds the app, wires CORS and routers."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return app


app = create_app()

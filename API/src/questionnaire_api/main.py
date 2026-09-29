"""
Создаёт HTTP-приложение ML-сервиса анкетирования.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from questionnaire_api.authentication import (
    ServiceAuthenticationMiddleware,
    ServiceTokenAuthenticator,
    install_service_authenticator,
)
from questionnaire_api.config import Settings, get_settings
from questionnaire_api.dependencies import (
    LLMClientFactory,
    TurnControllerFactory,
    create_llm_client,
    create_turn_controller,
)
from questionnaire_api.exception_handlers import (
    handle_request_validation_error,
    handle_unexpected_exception,
)
from questionnaire_api.logging_config import configure_logging
from questionnaire_api.request_body_limit import RequestBodyLimitMiddleware
from questionnaire_api.routes import turns_router
from questionnaire_api.runtime import ApplicationServices, get_application_services


class HealthResponse(BaseModel):
    """
    Описывает результат проверки работоспособности сервиса.
    """

    status: Literal["ok", "ready", "not_ready"]


def create_app(
    settings: Settings | None = None,
    llm_client_factory: LLMClientFactory = create_llm_client,
    turn_controller_factory: TurnControllerFactory = create_turn_controller,
) -> FastAPI:
    """
    Собирает приложение и связывает ресурсы с его ASGI lifespan.
    """
    active_settings = settings if settings is not None else get_settings()
    configure_logging(active_settings.log_level)
    service_authenticator = ServiceTokenAuthenticator(
        active_settings.service_auth_token,
        active_settings.service_auth_previous_token,
    )
    documentation_enabled = active_settings.app_env != "prod"

    @asynccontextmanager
    async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
        """
        Создаёт LLM-клиент при старте и гарантированно закрывает при остановке.
        """
        llm_client = llm_client_factory(active_settings) if active_settings.llm_configured else None
        try:
            turn_controller = (
                turn_controller_factory(llm_client) if llm_client is not None else None
            )
            application.state.services = ApplicationServices(
                llm_client=llm_client,
                turn_controller=turn_controller,
            )
            yield
        finally:
            application.state.services = None
            if llm_client is not None:
                await llm_client.aclose()

    application = FastAPI(
        title="Aisto Demo AI Service",
        version="0.1.0",
        lifespan=application_lifespan,
        openapi_url="/openapi.json" if documentation_enabled else None,
        docs_url="/docs" if documentation_enabled else None,
        redoc_url="/redoc" if documentation_enabled else None,
    )
    application.state.services = None
    install_service_authenticator(application, service_authenticator)
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_request_bytes=active_settings.api_max_request_bytes,
    )
    application.add_middleware(
        ServiceAuthenticationMiddleware,
        authenticator=service_authenticator,
    )
    application.add_exception_handler(
        RequestValidationError,
        handle_request_validation_error,
    )
    application.add_exception_handler(Exception, handle_unexpected_exception)
    application.include_router(turns_router)

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def check_liveness() -> HealthResponse:
        """
        Подтверждает, что процесс приложения запущен.
        """
        return HealthResponse(status="ok")

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        tags=["health"],
    )
    async def check_readiness(response: Response) -> HealthResponse:
        """
        Подтверждает завершение инициализации runtime-ресурсов приложения.
        """
        try:
            get_application_services(application)
        except RuntimeError:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(status="not_ready")
        return HealthResponse(status="ready")

    return application

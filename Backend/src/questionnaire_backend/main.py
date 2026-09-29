"""
Собирает отдельное FastAPI-приложение браузерного backend-слоя.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from questionnaire_backend.config import BackendSettings, get_backend_settings
from questionnaire_backend.errors import (
    DialogModeAlreadySelectedError,
    DialogModeNotSelectedError,
    MessageIdConflictError,
    MLProtocolError,
    MLTransportError,
    MLTurnFailureError,
    SessionCapacityError,
    SessionCompletedError,
    SessionNotFoundError,
    StateVersionConflictError,
)
from questionnaire_backend.logging_config import configure_logging
from questionnaire_backend.ml_client import MLClientFactory, create_http_ml_client
from questionnaire_backend.repository import Clock, InMemorySessionRepository, utc_now
from questionnaire_backend.request_body_limit import RequestBodyLimitMiddleware
from questionnaire_backend.routes import router
from questionnaire_backend.service import QuestionnaireService

_PUBLIC_ML_FAILURES: dict[str, tuple[str, bool]] = {
    "LLM_UNKNOWN": ("Неизвестная ошибка клиента языковой модели.", False),
    "LLM_CONFIGURATION": ("Клиент языковой модели настроен некорректно.", False),
    "LLM_CLIENT_CLOSED": ("Клиент языковой модели уже закрыт.", False),
    "LLM_TIMEOUT": ("Языковая модель не ответила за отведённое время.", True),
    "LLM_CONNECTION": ("Не удалось подключиться к провайдеру языковой модели.", True),
    "LLM_AUTHENTICATION": ("Провайдер языковой модели отклонил доступ.", False),
    "LLM_RATE_LIMIT": ("Провайдер языковой модели временно ограничил запросы.", True),
    "LLM_REQUEST_REJECTED": ("Провайдер отклонил параметры вызова.", False),
    "LLM_PROVIDER_UNAVAILABLE": ("Провайдер языковой модели временно недоступен.", True),
    "LLM_EMPTY_RESPONSE": ("Языковая модель вернула пустой ответ.", True),
    "LLM_TRUNCATED_RESPONSE": ("Ответ языковой модели был оборван.", True),
    "LLM_REFUSAL": ("Языковая модель отказалась сформировать ответ.", False),
    "LLM_INVALID_RESPONSE": ("Провайдер вернул некорректный ответ.", True),
    "AGENT_CONTEXT_TOO_LARGE": ("Контекст хода превышает допустимый размер.", False),
    "AGENT_OUTPUT_TOO_LARGE": ("Ответ языковой модели превышает допустимый размер.", True),
    "AGENT_OUTPUT_INVALID": ("Языковая модель вернула ответ неверного формата.", True),
    "AGENT_OUTPUT_REJECTED": ("Ответ модели не прошёл проверку правил анкеты.", True),
}


def create_app(
    settings: BackendSettings | None = None,
    ml_client_factory: MLClientFactory = create_http_ml_client,
    clock: Clock = utc_now,
) -> FastAPI:
    """
    Создаёт backend с серверным ML-клиентом и ограниченным хранилищем сессий.
    """
    active_settings = settings if settings is not None else get_backend_settings()
    configure_logging()

    @asynccontextmanager
    async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
        """
        Создаёт общие ресурсы при старте и закрывает ML-клиент при остановке.
        """
        ml_client = ml_client_factory(active_settings)
        repository = InMemorySessionRepository(
            ttl_seconds=active_settings.backend_session_ttl_seconds,
            max_sessions=active_settings.backend_max_sessions,
            clock=clock,
        )
        application.state.questionnaire_service = QuestionnaireService(
            repository,
            ml_client,
            clock=clock,
        )
        try:
            yield
        finally:
            application.state.questionnaire_service = None
            await ml_client.aclose()

    application = FastAPI(
        title="Questionnaire Application Backend",
        version="0.1.0",
        lifespan=application_lifespan,
    )
    application.state.questionnaire_service = None
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_request_bytes=active_settings.backend_max_request_bytes,
    )
    application.include_router(router)

    @application.get("/health/live", tags=["health"])
    async def check_liveness() -> dict[str, str]:
        """
        Подтверждает, что процесс backend отвечает на HTTP-запросы.
        """
        return {"status": "ok"}

    @application.get("/health/ready", tags=["health"])
    async def check_readiness() -> JSONResponse:
        """
        Проверяет инициализацию сессий и клиента без сетевого вызова ML.
        """
        ready = isinstance(application.state.questionnaire_service, QuestionnaireService)
        return JSONResponse(
            {"status": "ready" if ready else "not_ready"},
            status_code=200 if ready else 503,
            headers={"Cache-Control": "no-store"},
        )

    @application.middleware("http")
    async def disable_api_caching(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """
        Запрещает браузерное кеширование снимков и предметных ошибок API.
        """
        response = await call_next(request)
        if request.url.path == "/api" or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(
        request: Request,
        error: SessionNotFoundError,
    ) -> JSONResponse:
        """
        Возвращает одинаковый ответ для неизвестной и истёкшей сессии.
        """
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            "SESSION_NOT_FOUND",
            "Сессия не найдена или срок её действия истёк.",
        )

    @application.exception_handler(SessionCapacityError)
    async def handle_session_capacity(
        request: Request,
        error: SessionCapacityError,
    ) -> JSONResponse:
        """
        Сообщает о временной невозможности создать дополнительную сессию.
        """
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "SESSION_CAPACITY_REACHED",
            "Временно невозможно создать новую сессию.",
            retryable=True,
        )

    @application.exception_handler(StateVersionConflictError)
    async def handle_state_version_conflict(
        request: Request,
        error: StateVersionConflictError,
    ) -> JSONResponse:
        """
        Возвращает актуальную версию для восстановления клиента после конфликта.
        """
        return _error_response(
            status.HTTP_409_CONFLICT,
            "STATE_VERSION_CONFLICT",
            "Состояние сессии изменилось. Обновите данные и повторите действие.",
            currentStateVersion=error.current_version,
        )

    @application.exception_handler(MessageIdConflictError)
    async def handle_message_id_conflict(
        request: Request,
        error: MessageIdConflictError,
    ) -> JSONResponse:
        """
        Запрещает использовать один идемпотентный ключ для разных команд.
        """
        return _error_response(
            status.HTTP_409_CONFLICT,
            "MESSAGE_ID_CONFLICT",
            "Этот messageId уже использован для другой команды.",
        )

    @application.exception_handler(SessionCompletedError)
    async def handle_completed_session(
        request: Request,
        error: SessionCompletedError,
    ) -> JSONResponse:
        """
        Запрещает изменять уже завершённую анкету обычной репликой.
        """
        return _error_response(
            status.HTTP_409_CONFLICT,
            "SESSION_COMPLETED",
            "Анкета уже завершена.",
        )

    @application.exception_handler(DialogModeNotSelectedError)
    async def handle_dialog_mode_not_selected(
        request: Request,
        error: DialogModeNotSelectedError,
    ) -> JSONResponse:
        """
        Просит сначала выбрать один из двух поддерживаемых сценариев.
        """
        del request, error
        return _error_response(
            status.HTTP_409_CONFLICT,
            "DIALOG_MODE_REQUIRED",
            "Сначала выберите, как начать анкету.",
        )

    @application.exception_handler(DialogModeAlreadySelectedError)
    async def handle_dialog_mode_already_selected(
        request: Request,
        error: DialogModeAlreadySelectedError,
    ) -> JSONResponse:
        """
        Запрещает менять сценарий после начала содержательной части.
        """
        del request, error
        return _error_response(
            status.HTTP_409_CONFLICT,
            "DIALOG_MODE_ALREADY_SELECTED",
            "Сценарий начала анкеты уже выбран.",
        )

    @application.exception_handler(MLTransportError)
    async def handle_ml_transport_error(
        request: Request,
        error: MLTransportError,
    ) -> JSONResponse:
        """
        Скрывает сетевые детали и сообщает о повторяемом временном отказе.
        """
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ML_SERVICE_UNAVAILABLE",
            "Сервис обработки ответа временно недоступен.",
            retryable=True,
        )

    @application.exception_handler(MLProtocolError)
    async def handle_ml_protocol_error(
        request: Request,
        error: MLProtocolError,
    ) -> JSONResponse:
        """
        Скрывает повреждённый ответ зависимого сервиса.
        """
        return _error_response(
            status.HTTP_502_BAD_GATEWAY,
            "ML_RESPONSE_INVALID",
            "Сервис обработки ответа вернул некорректный результат.",
            retryable=True,
        )

    @application.exception_handler(MLTurnFailureError)
    async def handle_ml_turn_failure(
        request: Request,
        error: MLTurnFailureError,
    ) -> JSONResponse:
        """
        Передаёт безопасный контролируемый отказ без изменения состояния.
        """
        public_code, public_message, public_retryable = _public_ml_failure(error)
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if public_retryable
            else status.HTTP_422_UNPROCESSABLE_CONTENT
        )
        return _error_response(
            response_status,
            public_code,
            public_message,
            retryable=public_retryable,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        """
        Возвращает безопасную ошибку контракта без отражения входных значений.
        """
        del request, error
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "REQUEST_VALIDATION_FAILED",
            "HTTP-запрос не соответствует контракту.",
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        """
        Скрывает внутренние детали непредвиденной ошибки API.
        """
        del request, error
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_SERVER_ERROR",
            "Внутренняя ошибка сервиса.",
        )

    return application


def _error_response(
    response_status: int,
    code: str,
    message: str,
    **details: Any,
) -> JSONResponse:
    """
    Создаёт единый JSON-конверт безопасной публичной ошибки.
    """
    return JSONResponse(
        status_code=response_status,
        content={
            "detail": {"code": code, "message": message, **details},
            "message": message,
        },
        headers={"Cache-Control": "no-store"},
    )


def _public_ml_failure(error: MLTurnFailureError) -> tuple[str, str, bool]:
    """
    Преобразует только известный код ML в заранее заданное публичное сообщение.
    """
    known_failure = _PUBLIC_ML_FAILURES.get(error.code)
    if known_failure is None:
        return (
            "ML_TURN_FAILED",
            "Сервис обработки ответа временно не смог обработать реплику.",
            True,
        )
    message, retryable = known_failure
    return error.code, message, retryable

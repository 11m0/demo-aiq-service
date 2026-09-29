"""
Проверяет service-to-service Bearer-токен до чтения тела бизнес-запроса.
"""

from hashlib import sha256
from secrets import compare_digest
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from questionnaire_api.http_errors import APIErrorDetail, APIErrorResponse

_PROTECTED_API_PATH = "/v1"
_AUTHENTICATOR_STATE_ATTRIBUTE = "service_authenticator"
_AUTHENTICATION_ERROR_CODE = "SERVICE_AUTHENTICATION_FAILED"
_AUTHENTICATION_ERROR_MESSAGE = "Требуется действительный Bearer-токен сервиса."
_BEARER_CHALLENGE = "Bearer"
_API_UNAVAILABLE_CODE = "TURN_PROCESSING_UNAVAILABLE"
_API_UNAVAILABLE_MESSAGE = "Сервис обработки ходов временно недоступен."

_service_bearer = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    scheme_name="ServiceBearerAuth",
    description="Непрозрачный service-to-service токен backend-сервиса.",
)


class ServiceTokenAuthenticator:
    """
    Сверяет предъявленный токен с хешем настроенного секрета.
    """

    __slots__ = ("_expected_digests",)

    def __init__(
        self,
        token: SecretStr | None,
        previous_token: SecretStr | None = None,
    ) -> None:
        """
        Сохраняет только SHA-256 digest текущего и предыдущего секретов.
        """
        tokens = (token, previous_token)
        self._expected_digests = tuple(
            sha256(candidate.get_secret_value().encode("ascii")).digest()
            for candidate in tokens
            if candidate is not None
        )

    @property
    def enabled(self) -> bool:
        """
        Показывает, требуется ли токен для бизнес-запроса.
        """
        return bool(self._expected_digests)

    def accepts(self, token: str | None) -> bool:
        """
        Проверяет токен сравнением digest постоянной длины без нормализации.
        """
        if not self._expected_digests:
            return False
        if token is None:
            return False
        try:
            presented_digest = sha256(token.encode("ascii")).digest()
        except UnicodeEncodeError:
            return False
        comparison_results = tuple(
            compare_digest(presented_digest, expected_digest)
            for expected_digest in self._expected_digests
        )
        return any(comparison_results)


class ServiceAuthenticationMiddleware:
    """
    Отклоняет неавторизованный turn-запрос до чтения его HTTP body.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        authenticator: ServiceTokenAuthenticator,
    ) -> None:
        """
        Сохраняет следующее ASGI-приложение и неизменяемый проверяющий объект.
        """
        self._app = app
        self._authenticator = authenticator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Защищает бизнес-маршрут и пропускает остальные запросы без проверки.
        """
        if not self._must_authenticate(scope):
            await self._app(scope, receive, send)
            return

        async def send_without_cache(message: Message) -> None:
            """
            Запрещает HTTP-кеширование любого ответа версионированного API.
            """
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        if not self._authenticator.enabled:
            await _send_api_unavailable(scope, receive, send_without_cache)
            return

        token = _extract_bearer_token(scope)
        if not self._authenticator.accepts(token):
            await _send_authentication_error(scope, receive, send_without_cache)
            return
        await self._app(scope, receive, send_without_cache)

    @staticmethod
    def _must_authenticate(scope: Scope) -> bool:
        """
        Защищает всё версионированное пространство бизнес-API.
        """
        path = scope.get("path")
        return (
            scope.get("type") == "http"
            and isinstance(path, str)
            and (path == _PROTECTED_API_PATH or path.startswith(f"{_PROTECTED_API_PATH}/"))
        )


def install_service_authenticator(
    application: FastAPI,
    authenticator: ServiceTokenAuthenticator,
) -> None:
    """
    Устанавливает проверяющий объект в state приложения для FastAPI dependency.
    """
    application.state.service_authenticator = authenticator


def require_service_authentication(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(_service_bearer),
    ],
) -> None:
    """
    Повторно проверяет Bearer-токен на границе FastAPI и формирует OpenAPI security.
    """
    authenticator = getattr(request.app.state, _AUTHENTICATOR_STATE_ATTRIBUTE, None)
    token = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if isinstance(authenticator, ServiceTokenAuthenticator) and authenticator.accepts(token):
        return
    raise _authentication_http_exception()


def _extract_bearer_token(scope: Scope) -> str | None:
    """
    Извлекает ровно один ASCII Bearer-токен и отклоняет неоднозначные заголовки.
    """
    authorization_values = tuple(
        value for name, value in scope.get("headers", ()) if name.lower() == b"authorization"
    )
    if len(authorization_values) != 1:
        return None

    raw_scheme, separator, raw_token = authorization_values[0].partition(b" ")
    if raw_scheme.lower() != b"bearer" or not separator or not raw_token:
        return None
    try:
        return raw_token.decode("ascii")
    except UnicodeDecodeError:
        return None


def _authentication_http_exception() -> HTTPException:
    """
    Создаёт одинаковый 401-ответ для отсутствующего и неверного токена.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": _AUTHENTICATION_ERROR_CODE,
            "message": _AUTHENTICATION_ERROR_MESSAGE,
        },
        headers={"WWW-Authenticate": _BEARER_CHALLENGE},
    )


async def _send_authentication_error(
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """
    Возвращает безопасный 401 непосредственно на ASGI-границе.
    """
    response = APIErrorResponse(
        detail=APIErrorDetail(
            code=_AUTHENTICATION_ERROR_CODE,
            message=_AUTHENTICATION_ERROR_MESSAGE,
        )
    )
    await JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=response.model_dump(mode="json", by_alias=True),
        headers={"WWW-Authenticate": _BEARER_CHALLENGE},
    )(scope, receive, send)


async def _send_api_unavailable(
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """
    Закрывает `/v1` безопасным отказом, когда аутентификация не настроена.
    """
    response = APIErrorResponse(
        detail=APIErrorDetail(
            code=_API_UNAVAILABLE_CODE,
            message=_API_UNAVAILABLE_MESSAGE,
        )
    )
    await JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(mode="json", by_alias=True),
    )(scope, receive, send)

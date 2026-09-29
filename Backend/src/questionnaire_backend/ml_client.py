"""
Вызывает защищённый stateless ML API и проверяет его JSON-конверт.
"""

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from questionnaire_backend.config import BackendSettings
from questionnaire_backend.errors import MLProtocolError, MLTransportError
from questionnaire_contracts import TurnFailureResponse, TurnRequest, TurnResponse

_TURN_RESPONSE_ADAPTER = TypeAdapter(TurnResponse)
_LOGGER = logging.getLogger(__name__)


class MLTurnClient(Protocol):
    """
    Задаёт минимальный интерфейс клиента обработки одного хода.
    """

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """
        Отправляет полный ход и возвращает разобранный предметный ответ.
        """
        ...

    async def aclose(self) -> None:
        """
        Освобождает сетевые ресурсы клиента.
        """
        ...


class HttpMLTurnClient:
    """
    Отправляет Bearer-токен только по серверному соединению с ML API.
    """

    def __init__(
        self,
        *,
        turn_url: str,
        service_auth_token: str,
        timeout_seconds: float,
        max_response_bytes: int = 256 * 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Создаёт общий асинхронный HTTP-клиент без автоматических редиректов.
        """
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("Лимит ответа ML должен быть положительным целым числом")
        self._turn_url = turn_url
        self._max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {service_auth_token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=timeout_seconds,
                write=10.0,
                pool=10.0,
            ),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def process_turn(self, request: TurnRequest) -> TurnResponse:
        """
        Вызывает ML API и отвергает небезопасный или неожиданный ответ.
        """
        started = perf_counter()
        outcome = "error"
        error_code = "INTERNAL_SERVER_ERROR"
        http_status: int | str = "-"
        try:
            try:
                async with self._client.stream(
                    "POST",
                    self._turn_url,
                    json=request.model_dump(mode="json", by_alias=True),
                ) as response:
                    http_status = response.status_code
                    if response.status_code != httpx.codes.OK:
                        raise MLProtocolError
                    response_body = await _read_limited_response(
                        response,
                        self._max_response_bytes,
                    )
            except httpx.RequestError as error:
                raise MLTransportError from error

            try:
                turn_response = _TURN_RESPONSE_ADAPTER.validate_json(response_body)
            except ValidationError as error:
                raise MLProtocolError from error

            if isinstance(turn_response, TurnFailureResponse):
                outcome = "failure"
                code = turn_response.error.code
                error_code = (
                    code
                    if code.isascii() and len(code) <= 64 and code.replace("_", "").isalnum()
                    else "ML_TURN_FAILED"
                )
            else:
                outcome = "success"
                error_code = "-"
            return turn_response
        except MLTransportError:
            error_code = "ML_SERVICE_UNAVAILABLE"
            raise
        except MLProtocolError:
            error_code = "ML_RESPONSE_INVALID"
            raise
        finally:
            _LOGGER.info(
                "event=backend_ml_call correlation_id=%s duration_ms=%.1f "
                "http_status=%s outcome=%s error_code=%s",
                request.correlation_id,
                (perf_counter() - started) * 1000,
                http_status,
                outcome,
                error_code,
            )

    async def aclose(self) -> None:
        """
        Закрывает общий пул HTTP-соединений.
        """
        await self._client.aclose()


MLClientFactory = Callable[[BackendSettings], MLTurnClient]


def create_http_ml_client(settings: BackendSettings) -> MLTurnClient:
    """
    Создаёт рабочий HTTP-клиент из проверенных серверных настроек.
    """
    return HttpMLTurnClient(
        turn_url=settings.ml_turn_url,
        service_auth_token=settings.service_auth_token.get_secret_value(),
        timeout_seconds=settings.ml_timeout_seconds,
        max_response_bytes=settings.ml_response_max_bytes,
    )


async def _read_limited_response(
    response: httpx.Response,
    max_response_bytes: int,
) -> bytes:
    """
    Читает декодированный ответ ML-сервиса в пределах заданного размера.
    """
    raw_content_length = response.headers.get("content-length")
    if raw_content_length is not None:
        if (
            len(raw_content_length) > 20
            or not raw_content_length.isascii()
            or not raw_content_length.isdigit()
            or int(raw_content_length) > max_response_bytes
        ):
            raise MLProtocolError

    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > max_response_bytes:
            raise MLProtocolError
    return bytes(body)

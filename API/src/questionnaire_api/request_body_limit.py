"""
Ограничивает сырой HTTP body до разбора JSON и предметной валидации.
"""

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from questionnaire_api.http_errors import APIErrorDetail, APIErrorResponse

_TURN_ENDPOINT_PATH = "/v1/turns/process"


class RequestBodyLimitMiddleware:
    """
    Проверяет Content-Length и фактически прочитанные ASGI-чанки запроса.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        """
        Сохраняет положительный предел размера тела запроса.
        """
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes <= 0
        ):
            raise ValueError("Лимит HTTP body должен быть положительным целым числом")
        self._app = app
        self._max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Пропускает обычные запросы и отклоняет слишком большой turn body.
        """
        if not self._must_limit(scope):
            await self._app(scope, receive, send)
            return

        content_length = self._read_content_length(scope)
        if content_length == -1:
            await _send_error_response(
                scope,
                receive,
                send,
                status_code=400,
                code="INVALID_CONTENT_LENGTH",
                message="Заголовок Content-Length задан некорректно.",
            )
            return
        if content_length is not None and content_length > self._max_request_bytes:
            await self._send_too_large(scope, receive, send)
            return
        await self._buffer_with_limit(scope, receive, send)

    @staticmethod
    def _must_limit(scope: Scope) -> bool:
        """
        Ограничивает только тело бизнес-маршрута обработки хода.
        """
        return (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == _TURN_ENDPOINT_PATH
        )

    @staticmethod
    def _read_content_length(scope: Scope) -> int | None:
        """
        Возвращает длину, отсутствие заголовка либо признак повреждённого значения.
        """
        raw_value = Headers(scope=scope).get("content-length")
        if raw_value is None:
            return None
        if len(raw_value) > 20 or not raw_value.isascii() or not raw_value.isdigit():
            return -1
        try:
            return int(raw_value)
        except ValueError:
            return -1

    async def _buffer_with_limit(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """
        Буферизует допустимое тело и повторно передаёт его приложению.
        """
        received_bytes = 0
        buffered_messages: list[Message] = []

        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] != "http.request":
                break
            received_bytes += len(message.get("body", b""))
            if received_bytes > self._max_request_bytes:
                await self._send_too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        next_message_index = 0

        async def replay_receive() -> Message:
            """
            Воспроизводит сохранённые сообщения, затем возвращается к ASGI receive.
            """
            nonlocal next_message_index
            if next_message_index < len(buffered_messages):
                message = buffered_messages[next_message_index]
                next_message_index += 1
                return message
            return await receive()

        await self._app(scope, replay_receive, send)

    @staticmethod
    async def _send_too_large(scope: Scope, receive: Receive, send: Send) -> None:
        """
        Возвращает единый безопасный ответ о превышении размера body.
        """
        await _send_error_response(
            scope,
            receive,
            send,
            status_code=413,
            code="REQUEST_BODY_TOO_LARGE",
            message="Тело HTTP-запроса превышает допустимый размер.",
        )


async def _send_error_response(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    status_code: int,
    code: str,
    message: str,
) -> None:
    """
    Сериализует безопасную HTTP-ошибку напрямую на ASGI-границе.
    """
    response = APIErrorResponse(detail=APIErrorDetail(code=code, message=message))
    await JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode="json", by_alias=True),
    )(scope, receive, send)

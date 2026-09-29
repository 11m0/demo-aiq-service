"""
Ограничивает тело браузерных POST-запросов до разбора JSON.
"""

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """
    Проверяет заявленный и фактически полученный размер HTTP body.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        """
        Сохраняет положительный предел размера браузерного запроса.
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
        Пропускает запрос либо отклоняет его до JSON-декодирования.
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
        Выбирает только поддерживаемые POST-маршруты с JSON-телом.
        """
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return False
        path = scope.get("path")
        return path == "/api/sessions" or (
            isinstance(path, str)
            and path.startswith("/api/sessions/")
            and path.endswith("/messages")
        )

    @staticmethod
    def _read_content_length(scope: Scope) -> int | None:
        """
        Возвращает длину, отсутствие заголовка либо признак повреждения.
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
        Буферизует допустимые ASGI-чанки и воспроизводит их приложению.
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
            Возвращает сохранённые чанки, затем продолжает исходный поток.
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
        Возвращает безопасный ответ о превышении предела body.
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
    Отправляет единый публичный конверт ошибки на ASGI-границе.
    """
    await JSONResponse(
        status_code=status_code,
        content={
            "detail": {"code": code, "message": message},
            "message": message,
        },
        headers={"Cache-Control": "no-store"},
    )(scope, receive, send)

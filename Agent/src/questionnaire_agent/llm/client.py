"""
Задаёт независимый от провайдера контракт асинхронного LLM-клиента.
"""
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class LLMMessageRole(StrEnum):
    """
    Перечисляет роли обычных текстовых сообщений языковой модели.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """
    Представляет одно сообщение в полном контексте вызова модели.
    """
    role: LLMMessageRole
    content: str

    def __post_init__(self) -> None:
        """
        Запрещает отправку модели пустого текстового сообщения.
        """
        if not self.content.strip():
            raise ValueError("Сообщение языковой модели не должно быть пустым")


@dataclass(frozen=True, slots=True, init=False)
class LLMResponseSchema:
    """
    Описывает обязательную JSON Schema структурированного ответа модели.
    """
    name: str
    strict: bool
    _schema_json: str = field(repr=False)

    def __init__(self, name: str, schema: dict[str, Any], strict: bool = True) -> None:
        """
        Проверяет и сохраняет независимую JSON-копию схемы ответа.
        """
        if not name.strip():
            raise ValueError("Имя схемы ответа не должно быть пустым")
        if not schema:
            raise ValueError("JSON Schema ответа не должна быть пустой")

        try:
            schema_json = json.dumps(
                schema,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("JSON Schema ответа должна содержать только JSON-значения") from error

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "strict", strict)
        object.__setattr__(self, "_schema_json", schema_json)

    @property
    def schema(self) -> dict[str, Any]:
        """
        Возвращает новую изменяемую копию схемы для конкретного адаптера.
        """
        schema: dict[str, Any] = json.loads(self._schema_json)
        return schema


@dataclass(frozen=True, slots=True)
class LLMCompletionRequest:
    """
    Передаёт полный контекст одного stateless-вызова языковой модели.
    """
    messages: tuple[LLMMessage, ...]
    response_schema: LLMResponseSchema

    def __post_init__(self) -> None:
        """
        Требует хотя бы одно сообщение для вызова модели.
        """
        if not self.messages:
            raise ValueError("Вызов языковой модели должен содержать сообщения")


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """
    Хранит доступную статистику входных, выходных и кэшированных токенов.
    """
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        """
        Запрещает отрицательные значения счётчиков токенов.
        """
        token_counts = (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
        )
        if any(count is not None and count < 0 for count in token_counts):
            raise ValueError("Количество токенов не может быть отрицательным")
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("Кэшированных токенов не может быть больше входных")


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    """
    Возвращает сырой ответ модели и безопасные технические метаданные.
    """
    content: str
    model: str | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    usage: LLMTokenUsage | None = None

    def __post_init__(self) -> None:
        """
        Проверяет непустое содержимое и заполненные строковые метаданные.
        """
        if not self.content.strip():
            raise ValueError("Ответ языковой модели не должен быть пустым")

        optional_values = {
            "model": self.model,
            "finishReason": self.finish_reason,
            "providerRequestId": self.provider_request_id,
        }
        for field_name, value in optional_values.items():
            if value is not None and not value.strip():
                raise ValueError(f"Поле {field_name} не должно быть пустой строкой")


class LLMClient(Protocol):
    """
    Определяет минимальную асинхронную границу любого LLM-провайдера.
    """
    async def complete(self, request: LLMCompletionRequest) -> LLMCompletion:
        """
        Выполняет один структурированный вызов без сохранения состояния диалога.
        """
        ...

    async def aclose(self) -> None:
        """
        Освобождает сетевые ресурсы клиента идемпотентным способом.
        """
        ...

"""
Реализует stateless-адаптер OpenAI Responses API для LLM-контракта агента.
"""

import asyncio
import math
from typing import Any, Protocol

import openai

from questionnaire_agent.llm.client import (
    LLMCompletion,
    LLMCompletionRequest,
    LLMTokenUsage,
)
from questionnaire_agent.llm.errors import (
    LLMAuthenticationError,
    LLMClientClosedError,
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMTruncatedResponseError,
)


class _ResponsesResource(Protocol):
    """
    Описывает используемую часть ресурса Responses из OpenAI SDK.
    """

    async def create(self, **kwargs: Any) -> Any:
        """
        Создаёт один не потоковый ответ модели.
        """
        ...


class _AsyncOpenAIClient(Protocol):
    """
    Описывает минимальную поверхность внедряемого асинхронного SDK-клиента.
    """

    responses: _ResponsesResource
    max_retries: int

    async def close(self) -> None:
        """
        Закрывает сетевые ресурсы SDK-клиента.
        """
        ...


class OpenAIResponsesClient:
    """
    Выполняет один вызов Responses API без скрытой истории и повторов SDK.
    """

    _NONTERMINAL_STATUSES = frozenset({"cancelled", "in_progress", "queued"})
    _ALLOWED_OUTPUT_ITEM_TYPES = frozenset({"message", "reasoning"})
    _PROVIDER_UNAVAILABLE_CODES = frozenset({"server_error", "vector_store_timeout"})
    _REFUSAL_CODES = frozenset(
        {
            "content_filter",
            "content_policy_violation",
            "safety_violation",
        }
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        sdk_client: _AsyncOpenAIClient | None = None,
    ) -> None:
        """
        Проверяет настройки и создаёт SDK-клиент без автоматических повторов.
        """
        self._validate_configuration(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            sdk_client=sdk_client,
        )

        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._closed = False
        self._closing = False
        self._close_lock = asyncio.Lock()

        if sdk_client is not None:
            self._sdk_client = sdk_client
            return

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": timeout_seconds,
        }
        if base_url is not None:
            client_options["base_url"] = base_url

        configuration_error: LLMConfigurationError
        try:
            sdk_client = openai.AsyncOpenAI(**client_options)
        except (openai.OpenAIError, TypeError, ValueError):
            configuration_error = LLMConfigurationError()
        else:
            self._sdk_client = sdk_client
            return

        raise configuration_error

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletion:
        """
        Отправляет контекст и преобразует ответ SDK в провайдер-независимый результат.
        """
        if self._closed or self._closing:
            raise LLMClientClosedError()

        schema = request.response_schema
        translated_error: LLMClientError
        try:
            response = await self._sdk_client.responses.create(
                model=self._model,
                input=[
                    {"role": message.role.value, "content": message.content}
                    for message in request.messages
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema.name,
                        "schema": schema.schema,
                        "strict": schema.strict,
                    }
                },
                max_output_tokens=self._max_output_tokens,
                temperature=self._temperature,
                store=False,
                stream=False,
                truncation="disabled",
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            translated_error = LLMTimeoutError()
        except openai.APIError as error:
            translated_error = self._translate_sdk_error(error)
        else:
            return self._build_completion(response)

        raise translated_error

    async def aclose(self) -> None:
        """
        Закрывает SDK-клиент не более одного раза.
        """
        async with self._close_lock:
            if self._closed:
                return

            self._closing = True
            try:
                await self._sdk_client.close()
            except BaseException:
                self._closing = False
                raise
            self._closed = True
            self._closing = False

    @staticmethod
    def _validate_configuration(
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
        sdk_client: _AsyncOpenAIClient | None,
    ) -> None:
        """
        Отклоняет неоднозначные или неполные параметры до первого вызова сети.
        """
        if not isinstance(model, str) or not model or model != model.strip():
            raise LLMConfigurationError()
        if sdk_client is None and (
            not isinstance(api_key, str) or not api_key or api_key != api_key.strip()
        ):
            raise LLMConfigurationError()
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key or api_key != api_key.strip()
        ):
            raise LLMConfigurationError()
        if base_url is not None and (
            not isinstance(base_url, str) or not base_url or base_url != base_url.strip()
        ):
            raise LLMConfigurationError()
        if sdk_client is not None:
            max_retries = getattr(sdk_client, "max_retries", None)
            if (
                isinstance(max_retries, bool)
                or not isinstance(max_retries, int)
                or max_retries != 0
            ):
                raise LLMConfigurationError()
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or timeout_seconds <= 0
            or not math.isfinite(timeout_seconds)
        ):
            raise LLMConfigurationError()
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise LLMConfigurationError()
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise LLMConfigurationError()

    @classmethod
    def _translate_sdk_error(cls, error: openai.APIError) -> LLMClientError:
        """
        Переводит исключение SDK без копирования его текста и тела ответа.
        """
        status_code = cls._safe_status_code(getattr(error, "status_code", None))
        request_id = cls._safe_identifier(getattr(error, "request_id", None))
        metadata = {
            "provider_status_code": status_code,
            "provider_request_id": request_id,
        }

        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError(**metadata)
        if isinstance(error, openai.APIConnectionError):
            return LLMConnectionError(**metadata)
        if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return LLMAuthenticationError(**metadata)
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError(**metadata)
        if isinstance(error, openai.InternalServerError):
            return LLMProviderUnavailableError(**metadata)
        if isinstance(error, openai.APIResponseValidationError):
            return LLMInvalidResponseError(**metadata)
        if isinstance(error, openai.APIStatusError):
            if status_code in {401, 403}:
                return LLMAuthenticationError(**metadata)
            if status_code == 408:
                return LLMTimeoutError(**metadata)
            if status_code == 409:
                return LLMProviderUnavailableError(**metadata)
            if status_code == 429:
                return LLMRateLimitError(**metadata)
            if status_code is not None and status_code >= 500:
                return LLMProviderUnavailableError(**metadata)
            return LLMRequestRejectedError(**metadata)
        return LLMClientError(**metadata)

    def _build_completion(self, response: Any) -> LLMCompletion:
        """
        Проверяет терминальный ответ и извлекает только ожидаемый текстовый результат.
        """
        request_id = self._safe_identifier(getattr(response, "_request_id", None))
        response_error = getattr(response, "error", None)
        if response_error is not None:
            raise self._translate_response_error(response_error, request_id=request_id)

        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        if status is not None and not isinstance(status, str):
            raise LLMInvalidResponseError(provider_request_id=request_id)
        if status == "incomplete":
            reason = getattr(incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                raise LLMTruncatedResponseError(provider_request_id=request_id)
            if reason == "content_filter":
                raise LLMRefusalError(provider_request_id=request_id)
            raise LLMInvalidResponseError(provider_request_id=request_id)
        if status == "failed" or status in self._NONTERMINAL_STATUSES:
            raise LLMProviderUnavailableError(provider_request_id=request_id)
        if status not in {None, "completed"}:
            raise LLMInvalidResponseError(provider_request_id=request_id)
        if incomplete_details is not None:
            raise LLMInvalidResponseError(provider_request_id=request_id)

        output = getattr(response, "output", None)
        if not isinstance(output, list):
            raise LLMInvalidResponseError(provider_request_id=request_id)
        if self._contains_refusal(output):
            raise LLMRefusalError(provider_request_id=request_id)

        content = self._collect_output_text(output, request_id=request_id)
        if not content.strip():
            raise LLMEmptyResponseError(provider_request_id=request_id)

        return LLMCompletion(
            content=content,
            model=self._response_model(response),
            provider_request_id=request_id,
            usage=self._response_usage(response, request_id=request_id),
        )

    @classmethod
    def _translate_response_error(
        cls,
        response_error: Any,
        *,
        request_id: str | None,
    ) -> LLMClientError:
        """
        Классифицирует ошибку из тела Responses API по стабильному машинному коду.
        """
        code = getattr(response_error, "code", None)
        metadata = {"provider_request_id": request_id}
        if not isinstance(code, str):
            return LLMInvalidResponseError(**metadata)
        if code == "rate_limit_exceeded":
            return LLMRateLimitError(**metadata)
        if code in cls._PROVIDER_UNAVAILABLE_CODES:
            return LLMProviderUnavailableError(**metadata)
        if code in cls._REFUSAL_CODES:
            return LLMRefusalError(**metadata)
        return LLMRequestRejectedError(**metadata)

    @staticmethod
    def _contains_refusal(output: list[Any]) -> bool:
        """
        Ищет отказ во всех сообщениях раньше обработки возможного частичного текста.
        """
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            if any(getattr(block, "type", None) == "refusal" for block in content):
                return True
        return False

    def _collect_output_text(
        self,
        output: list[Any],
        *,
        request_id: str | None,
    ) -> str:
        """
        Объединяет текстовые блоки в порядке, установленном Responses API.
        """
        text_parts: list[str] = []
        for item in output:
            item_type = getattr(item, "type", None)
            if not isinstance(item_type, str) or item_type not in self._ALLOWED_OUTPUT_ITEM_TYPES:
                raise LLMInvalidResponseError(provider_request_id=request_id)
            if item_type == "reasoning":
                continue

            message_status = getattr(item, "status", None)
            if message_status is not None and not isinstance(message_status, str):
                raise LLMInvalidResponseError(provider_request_id=request_id)
            if message_status not in {None, "completed"}:
                raise LLMInvalidResponseError(provider_request_id=request_id)
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                raise LLMInvalidResponseError(provider_request_id=request_id)
            for block in content:
                if getattr(block, "type", None) != "output_text":
                    raise LLMInvalidResponseError(provider_request_id=request_id)
                text = getattr(block, "text", None)
                if not isinstance(text, str):
                    raise LLMInvalidResponseError(provider_request_id=request_id)
                text_parts.append(text)
        return "".join(text_parts)

    def _response_model(self, response: Any) -> str:
        """
        Возвращает объявленную провайдером модель либо настроенную модель запроса.
        """
        response_model = getattr(response, "model", None)
        if isinstance(response_model, str) and response_model.strip():
            return response_model
        return self._model

    @classmethod
    def _response_usage(
        cls,
        response: Any,
        *,
        request_id: str | None,
    ) -> LLMTokenUsage | None:
        """
        Преобразует корректные счётчики токенов и отклоняет повреждённые значения.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if not cls._is_token_count(input_tokens) or not cls._is_token_count(output_tokens):
            raise LLMInvalidResponseError(provider_request_id=request_id)
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", None)
        if not cls._is_token_count(cached_tokens) or cached_tokens > input_tokens:
            cached_tokens = None
        return LLMTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_tokens,
        )

    @staticmethod
    def _is_token_count(value: Any) -> bool:
        """
        Проверяет неотрицательное целое значение счётчика токенов.
        """
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _safe_status_code(value: Any) -> int | None:
        """
        Пропускает только допустимый HTTP-статус без преобразования типов.
        """
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
        return None

    @staticmethod
    def _safe_identifier(value: Any) -> str | None:
        """
        Пропускает короткий однострочный идентификатор без управляющих символов.
        """
        if not isinstance(value, str) or value != value.strip() or not 1 <= len(value) <= 256:
            return None
        if not value.isprintable():
            return None
        return value

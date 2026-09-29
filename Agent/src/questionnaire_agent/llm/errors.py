"""
Описывает безопасные типизированные ошибки взаимодействия с LLM-провайдером.
"""

from enum import StrEnum
from typing import ClassVar


class LLMErrorCode(StrEnum):
    """
    Перечисляет стабильные машинные коды сбоев LLM-клиента.
    """

    UNKNOWN = "LLM_UNKNOWN"
    CONFIGURATION = "LLM_CONFIGURATION"
    CLIENT_CLOSED = "LLM_CLIENT_CLOSED"
    TIMEOUT = "LLM_TIMEOUT"
    CONNECTION = "LLM_CONNECTION"
    AUTHENTICATION = "LLM_AUTHENTICATION"
    RATE_LIMIT = "LLM_RATE_LIMIT"
    REQUEST_REJECTED = "LLM_REQUEST_REJECTED"
    PROVIDER_UNAVAILABLE = "LLM_PROVIDER_UNAVAILABLE"
    EMPTY_RESPONSE = "LLM_EMPTY_RESPONSE"
    TRUNCATED_RESPONSE = "LLM_TRUNCATED_RESPONSE"
    REFUSAL = "LLM_REFUSAL"
    INVALID_RESPONSE = "LLM_INVALID_RESPONSE"


class LLMClientError(RuntimeError):
    """
    Содержит безопасное описание сбоя без промпта и тела ответа провайдера.
    """

    code: ClassVar[LLMErrorCode] = LLMErrorCode.UNKNOWN
    retryable: ClassVar[bool] = False
    default_message: ClassVar[str] = "Неизвестная ошибка клиента языковой модели."

    def __init__(
        self,
        *,
        provider_status_code: int | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        """
        Сохраняет только безопасный текст и ограниченные метаданные провайдера.
        """
        if not self.default_message.strip():
            raise ValueError("Безопасное сообщение об ошибке LLM-клиента не задано")
        if provider_status_code is not None and not 100 <= provider_status_code <= 599:
            raise ValueError("HTTP-статус провайдера должен находиться в диапазоне 100..599")
        if provider_request_id is not None and not provider_request_id.strip():
            raise ValueError("Идентификатор запроса провайдера не должен быть пустым")

        self.provider_status_code = provider_status_code
        self.provider_request_id = provider_request_id
        super().__init__(self.default_message)


class LLMConfigurationError(LLMClientError):
    """
    Сообщает об отсутствующей или некорректной конфигурации клиента.
    """

    code = LLMErrorCode.CONFIGURATION
    retryable = False
    default_message = "Клиент языковой модели настроен некорректно."


class LLMClientClosedError(LLMClientError):
    """
    Запрещает новый вызов после закрытия ресурсов клиента.
    """

    code = LLMErrorCode.CLIENT_CLOSED
    retryable = False
    default_message = "Клиент языковой модели уже закрыт."


class LLMTimeoutError(LLMClientError):
    """
    Сообщает о превышении времени ожидания ответа модели.
    """

    code = LLMErrorCode.TIMEOUT
    retryable = True
    default_message = "Языковая модель не ответила за отведённое время."


class LLMConnectionError(LLMClientError):
    """
    Сообщает о сетевой недоступности LLM-провайдера.
    """

    code = LLMErrorCode.CONNECTION
    retryable = True
    default_message = "Не удалось подключиться к провайдеру языковой модели."


class LLMAuthenticationError(LLMClientError):
    """
    Сообщает об отклонённых учётных данных или правах доступа.
    """

    code = LLMErrorCode.AUTHENTICATION
    retryable = False
    default_message = "Провайдер языковой модели отклонил доступ."


class LLMRateLimitError(LLMClientError):
    """
    Сообщает о временном превышении лимита запросов провайдера.
    """

    code = LLMErrorCode.RATE_LIMIT
    retryable = True
    default_message = "Провайдер языковой модели временно ограничил запросы."


class LLMRequestRejectedError(LLMClientError):
    """
    Сообщает об окончательном отклонении параметров вызова провайдером.
    """

    code = LLMErrorCode.REQUEST_REJECTED
    retryable = False
    default_message = "Провайдер отклонил параметры вызова языковой модели."


class LLMProviderUnavailableError(LLMClientError):
    """
    Сообщает о временной внутренней ошибке LLM-провайдера.
    """

    code = LLMErrorCode.PROVIDER_UNAVAILABLE
    retryable = True
    default_message = "Провайдер языковой модели временно недоступен."


class LLMEmptyResponseError(LLMClientError):
    """
    Сообщает об отсутствии содержимого в успешном ответе провайдера.
    """

    code = LLMErrorCode.EMPTY_RESPONSE
    retryable = True
    default_message = "Языковая модель вернула пустой ответ."


class LLMTruncatedResponseError(LLMClientError):
    """
    Сообщает о завершении ответа из-за ограничения его длины.
    """

    code = LLMErrorCode.TRUNCATED_RESPONSE
    retryable = True
    default_message = "Ответ языковой модели был оборван из-за ограничения длины."


class LLMRefusalError(LLMClientError):
    """
    Сообщает об отказе модели сформировать запрошенный структурированный ответ.
    """

    code = LLMErrorCode.REFUSAL
    retryable = False
    default_message = "Языковая модель отказалась сформировать ответ."


class LLMInvalidResponseError(LLMClientError):
    """
    Сообщает о некорректной структуре ответа LLM-провайдера.
    """

    code = LLMErrorCode.INVALID_RESPONSE
    retryable = True
    default_message = "Провайдер языковой модели вернул некорректный ответ."

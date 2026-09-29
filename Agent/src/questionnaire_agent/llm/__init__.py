"""
Объединяет адаптеры и протоколы взаимодействия с языковой моделью.
"""

from questionnaire_agent.llm.client import (
    LLMClient,
    LLMCompletion,
    LLMCompletionRequest,
    LLMMessage,
    LLMMessageRole,
    LLMResponseSchema,
    LLMTokenUsage,
)
from questionnaire_agent.llm.errors import (
    LLMAuthenticationError,
    LLMClientClosedError,
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMErrorCode,
    LLMInvalidResponseError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMTruncatedResponseError,
)
from questionnaire_agent.llm.fake_client import FakeLLMClient
from questionnaire_agent.llm.openai_responses_client import OpenAIResponsesClient

__all__ = [
    "FakeLLMClient",
    "LLMAuthenticationError",
    "LLMClient",
    "LLMClientClosedError",
    "LLMClientError",
    "LLMCompletion",
    "LLMCompletionRequest",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMEmptyResponseError",
    "LLMErrorCode",
    "LLMInvalidResponseError",
    "LLMMessage",
    "LLMMessageRole",
    "LLMProviderUnavailableError",
    "LLMRateLimitError",
    "LLMRefusalError",
    "LLMRequestRejectedError",
    "LLMResponseSchema",
    "LLMTimeoutError",
    "LLMTokenUsage",
    "LLMTruncatedResponseError",
    "OpenAIResponsesClient",
]

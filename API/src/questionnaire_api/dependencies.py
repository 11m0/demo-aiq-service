"""
Создаёт внешние зависимости HTTP-приложения из проверенных настроек.
"""

from collections.abc import Callable

from fastapi import HTTPException, Request, status

from questionnaire_agent.llm import (
    LLMClient,
    LLMConfigurationError,
    OpenAIResponsesClient,
)
from questionnaire_agent.orchestration import QuestionnaireTurnController
from questionnaire_api.config import Settings
from questionnaire_api.runtime import get_application_services

LLMClientFactory = Callable[[Settings], LLMClient]
TurnControllerFactory = Callable[[LLMClient], QuestionnaireTurnController]


def create_llm_client(settings: Settings) -> LLMClient:
    """
    Создаёт OpenAI Responses-клиент из полной конфигурации приложения.
    """
    if not settings.llm_configured:
        raise LLMConfigurationError()

    api_key = settings.llm_api_key
    model = settings.llm_model
    if api_key is None or model is None:
        raise LLMConfigurationError()

    return OpenAIResponsesClient(
        api_key=api_key.get_secret_value(),
        model=model,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
    )


def create_turn_controller(llm_client: LLMClient) -> QuestionnaireTurnController:
    """
    Подключает генерацию цельной реплики к общему LLM-клиенту приложения.
    """
    return QuestionnaireTurnController(llm_client, reply_writer_client=llm_client)


def get_runtime_turn_controller(request: Request) -> QuestionnaireTurnController:
    """
    Возвращает готовый контроллер либо безопасный HTTP-отказ без обращения к сети.
    """
    try:
        services = get_application_services(request.app)
    except RuntimeError:
        services = None

    if services is None or services.turn_controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TURN_PROCESSING_UNAVAILABLE",
                "message": "Сервис обработки ходов временно недоступен.",
            },
        )
    return services.turn_controller

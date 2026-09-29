"""
Хранит типизированные ресурсы, принадлежащие жизненному циклу приложения.
"""

from dataclasses import dataclass

from fastapi import FastAPI

from questionnaire_agent.llm import LLMClient
from questionnaire_agent.orchestration import QuestionnaireTurnController


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """
    Объединяет созданные при старте внешние клиенты приложения.
    """

    llm_client: LLMClient | None
    turn_controller: QuestionnaireTurnController | None


def get_application_services(application: FastAPI) -> ApplicationServices:
    """
    Возвращает только полностью установленный контейнер runtime-ресурсов.
    """
    services = getattr(application.state, "services", None)
    if not isinstance(services, ApplicationServices):
        raise RuntimeError("Ресурсы приложения ещё не инициализированы")
    return services

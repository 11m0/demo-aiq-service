"""
Предоставляет программируемый LLM-клиент для тестов и локальных сценариев.
"""

from collections import deque
from collections.abc import Iterable

from questionnaire_agent.llm.client import LLMCompletion, LLMCompletionRequest
from questionnaire_agent.llm.errors import LLMClientClosedError, LLMClientError

FakeLLMOutcome = LLMCompletion | LLMClientError


class FakeLLMClient:
    """
    Возвращает заранее заданные результаты и записывает каждый запрос.
    """

    def __init__(self, outcomes: Iterable[FakeLLMOutcome] = ()) -> None:
        """
        Создаёт очередь результатов и ошибок в ожидаемом порядке вызовов.
        """
        self._outcomes = deque(outcomes)
        self._requests: list[LLMCompletionRequest] = []
        self._closed = False

    @property
    def requests(self) -> tuple[LLMCompletionRequest, ...]:
        """
        Возвращает неизменяемый снимок всех полученных запросов.
        """
        return tuple(self._requests)

    @property
    def closed(self) -> bool:
        """
        Показывает, были ли ресурсы fake-клиента закрыты.
        """
        return self._closed

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletion:
        """
        Записывает запрос и возвращает следующий результат тестового сценария.
        """
        if self._closed:
            raise LLMClientClosedError()

        self._requests.append(request)
        if not self._outcomes:
            raise AssertionError("FakeLLMClient получил незапланированный вызов")

        outcome = self._outcomes.popleft()
        if isinstance(outcome, LLMClientError):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        """
        Идемпотентно помечает fake-клиент закрытым.
        """
        self._closed = True

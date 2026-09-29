"""
Хранит локальные сессии с ограничением ёмкости и временем бездействия.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from questionnaire_backend.dialog import DialogMode
from questionnaire_backend.errors import SessionCapacityError, SessionNotFoundError
from questionnaire_contracts import (
    ActiveQuestionContext,
    DialogMessage,
    QuestionnaireDefinition,
    QuestionnaireState,
    RespondentContext,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """
    Возвращает текущее время UTC с информацией о часовом поясе.
    """
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ProcessedMessage:
    """
    Хранит отпечаток принятой идемпотентной команды и результат её применения.
    """

    content: str
    state_version: int
    applied: bool = False


@dataclass(frozen=True, slots=True)
class QuestionnaireSession:
    """
    Хранит полный серверный агрегат одной сессии анкетирования.
    """

    session_id: str
    respondent: RespondentContext
    dialog_mode: DialogMode
    questionnaire: QuestionnaireDefinition
    state: QuestionnaireState
    active_question: ActiveQuestionContext | None
    dialog_history: tuple[DialogMessage, ...]
    processed_messages: Mapping[str, ProcessedMessage]


@dataclass(slots=True)
class StoredSession:
    """
    Связывает агрегат с блокировкой и сроком жизни в памяти.
    """

    session: QuestionnaireSession
    lock: asyncio.Lock
    expires_at: datetime


class InMemorySessionRepository:
    """
    Последовательно обслуживает каждую сессию внутри одного backend-процесса.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_sessions: int,
        clock: Clock = utc_now,
    ) -> None:
        """
        Настраивает скользящий TTL, предел сессий и источник времени.
        """
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self._clock = clock
        self._records: dict[str, StoredSession] = {}
        self._guard = asyncio.Lock()

    async def add(self, session: QuestionnaireSession) -> None:
        """
        Добавляет новую сессию после удаления истёкших записей.
        """
        async with self._guard:
            now = self._clock()
            self._purge_expired_locked(now)
            if len(self._records) >= self._max_sessions:
                raise SessionCapacityError
            self._records[session.session_id] = StoredSession(
                session=session,
                lock=asyncio.Lock(),
                expires_at=now + self._ttl,
            )

    async def remove(self, session_id: str) -> None:
        """
        Удаляет сессию после завершения уже выполняющейся команды.
        """
        async with self._guard:
            now = self._clock()
            self._purge_expired_locked(now)
            record = self._records.get(session_id)
            if record is None:
                raise SessionNotFoundError

        await record.lock.acquire()
        try:
            async with self._guard:
                if self._records.get(session_id) is not record:
                    raise SessionNotFoundError
                del self._records[session_id]
        finally:
            record.lock.release()

    @asynccontextmanager
    async def locked(self, session_id: str) -> AsyncIterator[StoredSession]:
        """
        Выдаёт существующую сессию под её персональной блокировкой.
        """
        async with self._guard:
            now = self._clock()
            self._purge_expired_locked(now)
            record = self._records.get(session_id)
            if record is None:
                raise SessionNotFoundError
            record.expires_at = now + self._ttl

        await record.lock.acquire()
        try:
            yield record
        finally:
            record.expires_at = self._clock() + self._ttl
            record.lock.release()

    def _purge_expired_locked(self, now: datetime) -> None:
        """
        Удаляет истёкшие записи, которые сейчас не обрабатывают команду.
        """
        expired_ids = tuple(
            session_id
            for session_id, record in self._records.items()
            if record.expires_at <= now and not record.lock.locked()
        )
        for session_id in expired_ids:
            del self._records[session_id]

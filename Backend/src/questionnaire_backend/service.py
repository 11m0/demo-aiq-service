"""
Оркестрирует жизненный цикл сессии и атомарную обработку пользовательской реплики.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from uuid import uuid4

from questionnaire_backend.dialog import (
    OPENING_MESSAGE,
    DialogMode,
    dialog_mode_assistant_message,
    dialog_mode_user_message,
)
from questionnaire_backend.errors import (
    DialogModeAlreadySelectedError,
    DialogModeNotSelectedError,
    MessageIdConflictError,
    MLProtocolError,
    MLTransportError,
    MLTurnFailureError,
    SessionCompletedError,
    SessionNotFoundError,
    StateVersionConflictError,
)
from questionnaire_backend.ml_client import MLTurnClient
from questionnaire_backend.models import (
    CreateSessionRequest,
    PublicSessionSnapshot,
    SelectDialogModeRequest,
    SubmitMessageRequest,
)
from questionnaire_backend.presenter import present_session
from questionnaire_backend.questionnaire import (
    build_initial_questionnaire,
    build_initial_state,
)
from questionnaire_backend.repository import (
    Clock,
    InMemorySessionRepository,
    ProcessedMessage,
    QuestionnaireSession,
    utc_now,
)
from questionnaire_backend.state_machine import project_successful_turn
from questionnaire_contracts import (
    ActiveQuestionContext,
    AnswerCorrectionAuthorization,
    AnswerCorrectionMode,
    DialogMessage,
    DialogRole,
    QuestionnaireStatus,
    QuestionStatus,
    QuestionType,
    RespondentContext,
    TurnAnswerScope,
    TurnFailureResponse,
    TurnRequest,
    TurnSuccessResponse,
)
from questionnaire_core.validation import InvalidTurnResponseError, validate_turn_response

IdentifierFactory = Callable[[str], str]
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SubmitTiming:
    """
    Хранит служебные метки одного хода без содержимого реплики.
    """

    correlation_id: str | None = None
    lock_wait_ms: float | None = None
    outcome: str = "error"


def generate_identifier(prefix: str) -> str:
    """
    Создаёт непрогнозируемый идентификатор с читаемым назначением.
    """
    return f"{prefix}-{uuid4().hex}"


class QuestionnaireService:
    """
    Владеет сессиями и единственной точкой вызова stateless ML-сервиса.
    """

    def __init__(
        self,
        repository: InMemorySessionRepository,
        ml_client: MLTurnClient,
        *,
        clock: Clock = utc_now,
        identifier_factory: IdentifierFactory = generate_identifier,
    ) -> None:
        """
        Принимает хранилище, ML-клиент и детерминируемые системные зависимости.
        """
        self._repository = repository
        self._ml_client = ml_client
        self._clock = clock
        self._identifier_factory = identifier_factory

    async def create_session(
        self,
        request: CreateSessionRequest,
    ) -> PublicSessionSnapshot:
        """
        Создаёт сессию в ожидании выбора начального сценария без вызова модели.
        """
        questionnaire = build_initial_questionnaire()
        initial_state = build_initial_state(questionnaire)
        created_at = self._clock()
        greeting = DialogMessage(
            message_id=self._identifier_factory("assistant"),
            role=DialogRole.ASSISTANT,
            content=OPENING_MESSAGE,
            created_at=created_at,
        )
        session = QuestionnaireSession(
            session_id=self._identifier_factory("session"),
            respondent=RespondentContext(
                age=request.age,
                mode=request.respondent_mode,
                locale="ru-RU",
            ),
            dialog_mode=DialogMode.UNSELECTED,
            questionnaire=questionnaire,
            state=initial_state,
            active_question=None,
            dialog_history=(greeting,),
            processed_messages={},
        )
        snapshot = present_session(session)
        await self._repository.add(session)
        return snapshot

    async def select_dialog_mode(
        self,
        session_id: str,
        command: SelectDialogModeRequest,
    ) -> PublicSessionSnapshot:
        """
        Фиксирует выбор сценария и не обращается к ML-сервису.
        """
        fingerprint = ProcessedMessage(
            content=f"DIALOG_MODE:{command.dialog_mode.value}",
            state_version=command.state_version,
        )
        async with self._repository.locked(session_id) as stored:
            session = stored.session
            duplicate = _duplicate_command_result(
                session,
                command.message_id,
                fingerprint,
            )
            if duplicate is not None:
                return duplicate

            if any(message.message_id == command.message_id for message in session.dialog_history):
                raise MessageIdConflictError

            if session.state.status != QuestionnaireStatus.IN_PROGRESS:
                raise SessionCompletedError
            if command.state_version != session.state.state_version:
                raise StateVersionConflictError(session.state.state_version)
            if session.dialog_mode != DialogMode.UNSELECTED:
                raise DialogModeAlreadySelectedError

            created_at = self._clock()
            user_message = DialogMessage(
                message_id=command.message_id,
                role=DialogRole.USER,
                content=dialog_mode_user_message(command.dialog_mode),
                created_at=created_at,
            )
            assistant_message = DialogMessage(
                message_id=self._identifier_factory("assistant"),
                role=DialogRole.ASSISTANT,
                content=dialog_mode_assistant_message(
                    command.dialog_mode,
                    session.questionnaire,
                ),
                created_at=created_at,
            )
            active_question = None
            if command.dialog_mode == DialogMode.GUIDED:
                active_question = ActiveQuestionContext(
                    question_code=session.questionnaire.questions[0].question_code,
                    assistant_message_id=assistant_message.message_id,
                )

            processed_messages = dict(session.processed_messages)
            processed_messages[command.message_id] = replace(fingerprint, applied=True)
            updated_session = replace(
                session,
                dialog_mode=command.dialog_mode,
                state=session.state.model_copy(
                    update={"state_version": session.state.state_version + 1}
                ),
                active_question=active_question,
                dialog_history=(
                    *session.dialog_history,
                    user_message,
                    assistant_message,
                ),
                processed_messages=processed_messages,
            )
            stored.session = updated_session
            return present_session(updated_session)

    async def get_session(self, session_id: str) -> PublicSessionSnapshot:
        """
        Возвращает согласованный текущий снимок существующей сессии.
        """
        async with self._repository.locked(session_id) as stored:
            return present_session(stored.session)

    async def delete_session(self, session_id: str) -> None:
        """
        Освобождает серверные ресурсы больше не нужной браузерной сессии.
        """
        await self._repository.remove(session_id)

    async def submit_message(
        self,
        session_id: str,
        command: SubmitMessageRequest,
    ) -> PublicSessionSnapshot:
        """
        Измеряет полный ход, включая ожидание блокировки сессии.
        """
        started = perf_counter()
        timing = _SubmitTiming()
        error_code = "-"
        try:
            return await self._submit_message(session_id, command, timing)
        except Exception as error:
            error_code = _submit_error_code(error)
            raise
        finally:
            _LOGGER.info(
                "event=backend_turn_total correlation_id=%s duration_ms=%.1f "
                "lock_wait_ms=%s outcome=%s error_code=%s",
                timing.correlation_id or "-",
                (perf_counter() - started) * 1000,
                f"{timing.lock_wait_ms:.1f}" if timing.lock_wait_ms is not None else "-",
                timing.outcome,
                error_code,
            )

    async def _submit_message(
        self,
        session_id: str,
        command: SubmitMessageRequest,
        timing: _SubmitTiming,
    ) -> PublicSessionSnapshot:
        """
        Обрабатывает реплику последовательно и фиксирует только полный успех.
        """
        fingerprint = ProcessedMessage(
            content=command.content,
            state_version=command.state_version,
        )
        lock_started = perf_counter()
        async with self._repository.locked(session_id) as stored:
            timing.lock_wait_ms = (perf_counter() - lock_started) * 1000
            session = stored.session
            duplicate = _duplicate_command_result(
                session,
                command.message_id,
                fingerprint,
            )
            if duplicate is not None:
                timing.outcome = "duplicate"
                return duplicate

            if any(message.message_id == command.message_id for message in session.dialog_history):
                raise MessageIdConflictError

            if session.state.status != QuestionnaireStatus.IN_PROGRESS:
                raise SessionCompletedError
            if command.state_version != session.state.state_version:
                raise StateVersionConflictError(session.state.state_version)
            if session.dialog_mode == DialogMode.UNSELECTED:
                raise DialogModeNotSelectedError

            reserved_messages = dict(session.processed_messages)
            reserved_messages[command.message_id] = fingerprint
            session = replace(session, processed_messages=reserved_messages)
            stored.session = session

            user_message = DialogMessage(
                message_id=command.message_id,
                role=DialogRole.USER,
                content=command.content,
                created_at=self._clock(),
            )
            answer_scope = (
                TurnAnswerScope.ALL_UNANSWERED
                if session.dialog_mode == DialogMode.FREE_STORY
                and session.active_question is None
                else TurnAnswerScope.ACTIVE_QUESTION
            )
            answer_correction = _additive_e1_authorization(session, answer_scope)
            replaceable_question_codes = _replaceable_question_codes(session, answer_scope)
            turn_request = TurnRequest(
                schema_version="1.3.0",
                correlation_id=self._identifier_factory("request"),
                session_id=session.session_id,
                respondent=session.respondent,
                questionnaire=session.questionnaire,
                state=session.state,
                active_question=session.active_question,
                answer_correction=answer_correction,
                replaceable_question_codes=replaceable_question_codes,
                answer_scope=answer_scope,
                dialog_history=session.dialog_history,
                current_message=user_message,
                requested_at=self._clock(),
            )
            timing.correlation_id = turn_request.correlation_id
            response = await self._ml_client.process_turn(turn_request)
            if not _response_context_matches(turn_request, response):
                raise MLProtocolError
            if isinstance(response, TurnFailureResponse):
                raise MLTurnFailureError(
                    response.error.code,
                    response.error.message,
                    response.error.retryable,
                )
            if not isinstance(response, TurnSuccessResponse):
                raise MLProtocolError

            try:
                validate_turn_response(turn_request, response)
                assistant_message = DialogMessage(
                    message_id=self._identifier_factory("assistant"),
                    role=DialogRole.ASSISTANT,
                    content=response.assistant_reply,
                    created_at=self._clock(),
                )
                projection = project_successful_turn(
                    session,
                    user_message,
                    assistant_message,
                    response,
                    answer_correction=answer_correction,
                    replaceable_question_codes=replaceable_question_codes,
                )
                processed_messages = dict(session.processed_messages)
                processed_messages[command.message_id] = replace(
                    fingerprint,
                    applied=True,
                )
                updated_session = replace(
                    session,
                    state=projection.state,
                    active_question=projection.active_question,
                    dialog_history=projection.dialog_history,
                    processed_messages=processed_messages,
                )
                snapshot = present_session(updated_session)
            except (InvalidTurnResponseError, KeyError, ValueError) as error:
                raise MLProtocolError from error

            stored.session = updated_session
            timing.outcome = "success"
            return snapshot


def _submit_error_code(error: Exception) -> str:
    """
    Возвращает безопасный код отказа для журнала полного хода.
    """
    if isinstance(error, MLTurnFailureError):
        code = error.code
        if code.isascii() and len(code) <= 64 and code.replace("_", "").isalnum():
            return code
        return "ML_TURN_FAILED"
    error_codes = {
        SessionNotFoundError: "SESSION_NOT_FOUND",
        MessageIdConflictError: "MESSAGE_ID_CONFLICT",
        SessionCompletedError: "SESSION_COMPLETED",
        StateVersionConflictError: "STATE_VERSION_CONFLICT",
        DialogModeNotSelectedError: "DIALOG_MODE_REQUIRED",
        MLTransportError: "ML_SERVICE_UNAVAILABLE",
        MLProtocolError: "ML_RESPONSE_INVALID",
    }
    return error_codes.get(type(error), "INTERNAL_SERVER_ERROR")


def _replaceable_question_codes(
    session: QuestionnaireSession,
    answer_scope: TurnAnswerScope,
) -> tuple[str, ...]:
    """
    Разрешает пересмотр сохранённых одиночных ответов при другом активном вопросе.
    """
    active_question = session.active_question
    if answer_scope != TurnAnswerScope.ACTIVE_QUESTION or active_question is None:
        return ()

    states_by_code = {item.question_code: item for item in session.state.questions}
    return tuple(
        question.question_code
        for question in session.questionnaire.questions
        if question.question_code != active_question.question_code
        and question.question_type
        in {
            QuestionType.SCALE_0_3,
            QuestionType.PAIR_CHOICE,
            QuestionType.SINGLE_CHOICE,
        }
        and states_by_code[question.question_code].status == QuestionStatus.ANSWERED
    )


def _additive_e1_authorization(
    session: QuestionnaireSession,
    answer_scope: TurnAnswerScope,
) -> AnswerCorrectionAuthorization | None:
    """
    Разрешает дополнить сохранённые любимые занятия при другом активном вопросе.
    """
    active_question = session.active_question
    if (
        answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or active_question is None
        or active_question.question_code == "E1"
    ):
        return None

    e1_question = next(
        (question for question in session.questionnaire.questions if question.question_code == "E1"),
        None,
    )
    e1_state = next(
        (state for state in session.state.questions if state.question_code == "E1"),
        None,
    )
    if (
        e1_question is None
        or e1_question.question_type != QuestionType.MULTI_CHOICE
        or e1_state is None
        or e1_state.status not in {
            QuestionStatus.ANSWERED,
            QuestionStatus.PARTIALLY_ANSWERED,
        }
        or e1_state.answer is None
        or len(e1_state.answer.option_codes) >= e1_question.max_selections
    ):
        return None
    return AnswerCorrectionAuthorization(
        question_code="E1",
        mode=AnswerCorrectionMode.ADDITIVE,
    )


def _response_context_matches(
    request: TurnRequest,
    response: TurnSuccessResponse | TurnFailureResponse,
) -> bool:
    """
    Проверяет служебный конверт до обработки успеха или предметного отказа.
    """
    return (
        response.schema_version == request.schema_version
        and response.correlation_id == request.correlation_id
        and response.session_id == request.session_id
        and response.source_message_id == request.current_message.message_id
        and response.base_state_version == request.state.state_version
        and response.questionnaire_code == request.questionnaire.questionnaire_code
        and response.questionnaire_version == request.questionnaire.questionnaire_version
    )


def _duplicate_command_result(
    session: QuestionnaireSession,
    message_id: str,
    fingerprint: ProcessedMessage,
) -> PublicSessionSnapshot | None:
    """
    Возвращает результат применённой команды или проверяет конфликт ключа.
    """
    previous_fingerprint = session.processed_messages.get(message_id)
    if previous_fingerprint is None:
        return None
    if (
        previous_fingerprint.content != fingerprint.content
        or previous_fingerprint.state_version != fingerprint.state_version
    ):
        raise MessageIdConflictError
    if previous_fingerprint.applied:
        return present_session(session)
    return None

"""
Проверяет предложение агента в контексте исходного хода диалога.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from questionnaire_contracts import (
    AnswerCorrectionMode,
    ApplyDefaultAnswerChange,
    ConfirmAnswerChange,
    DialogMessage,
    DialogRole,
    NextActionType,
    PolicySkipReason,
    Question,
    QuestionState,
    QuestionStatus,
    QuestionType,
    RequestClarificationChange,
    ReviseMultiChoiceChange,
    SaveAnswerChange,
    SkipQuestionChange,
    StateChange,
    TurnAnswerScope,
    TurnRequest,
    TurnSuccessResponse,
)
from questionnaire_contracts.contract_types import (
    has_invalid_dialog_joiner_context,
    is_forbidden_dialog_text_character,
)
from questionnaire_core.answer_evidence import (
    is_explicit_skip_evidence,
    is_safe_semantic_answer_evidence,
)
from questionnaire_core.clarification import (
    is_safe_contextual_clarification,
    relevant_clarification_context_texts,
)
from questionnaire_core.dialog import (
    canonical_turn_reply,
    partial_multi_choice_followup,
    partial_multi_choice_prompt,
    repeats_official_question,
)

_SHORT_ANSWER_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
_PHONE_NUMBER_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_UNSAFE_TRANSITION_WORD_PATTERN = re.compile(
    (
        r"\b(?:адрес\w*|backend\w*|email\w*|ignore\w*|json\w*|llm\w*|"
        r"location\w*|password\w*|prompt\w*|system\w*|token\w*|"
        r"геолокац\w*|игнорир\w*|инструкц\w*|логин\w*|местополож\w*|"
        r"номер\w*|парол\w*|паспорт\w*|пинкод\w*|правил\w*|раскро\w*|"
        r"реквизит\w*|секрет\w*|снилс\w*|токен\w*)\b"
    ),
    flags=re.IGNORECASE,
)
_UNSAFE_TRANSITION_MARKERS = ("http://", "https://", "www.", "```", "<", ">")
_EXPLANATION_OPTION_REFERENCE_PATTERN = re.compile(r"\b[OV][1-9][0-9]*\b", re.IGNORECASE)
_UNSAFE_EXPLANATION_WORD_PATTERN = re.compile(
    r"\b(?:зовут|имя|фамили\w*|телефон\w*|назови\w*|напиши\w*|отправ\w*)\b",
    re.IGNORECASE,
)
_FRIENDLY_EXPLANATION_SYMBOLS = frozenset({"🙂", "😊"})
_MAX_TRANSITION_PREFIX_LENGTH = 120
_MAX_NATURAL_QUESTION_LENGTH = 240
_MECHANICAL_TRANSITION_PREFIXES = frozenset({"понял", "поняла", "ясно", "хорошо", "принято"})
_OPEN_ADDITIONAL_CHOICE_SUBJECT = re.compile(
    r"\b(?:что(?:-то|-нибудь)?|чем|каки\w*|заняти\w*|дел\w*|увлечени\w*|интерес\w*|нрав\w*)\b",
    flags=re.IGNORECASE,
)
_OPEN_ADDITIONAL_CHOICE_SCOPE = re.compile(
    r"\b(?:ещ[её]|друг\w*|кроме|помимо|осталь\w*|нов\w*)\b",
    flags=re.IGNORECASE,
)


class TurnResponseViolationCode(StrEnum):
    """
    Перечисляет стабильные машинные коды нарушений ответа агента.
    """

    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    UNKNOWN_QUESTION = "UNKNOWN_QUESTION"
    INVALID_OPTION = "INVALID_OPTION"
    INVALID_SELECTION_COUNT = "INVALID_SELECTION_COUNT"
    ADDITIVE_OPTION_CONFLICT = "ADDITIVE_OPTION_CONFLICT"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    QUESTION_NOT_ACTIVE = "QUESTION_NOT_ACTIVE"
    REPLACEMENT_NOT_AUTHORIZED = "REPLACEMENT_NOT_AUTHORIZED"
    UNKNOWN_SOURCE_MESSAGE = "UNKNOWN_SOURCE_MESSAGE"
    INVALID_SOURCE_ROLE = "INVALID_SOURCE_ROLE"
    CURRENT_MESSAGE_NOT_REFERENCED = "CURRENT_MESSAGE_NOT_REFERENCED"
    INVALID_SOURCE_SET = "INVALID_SOURCE_SET"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    UNSAFE_ANSWER_EVIDENCE = "UNSAFE_ANSWER_EVIDENCE"
    POLICY_SKIP_NOT_ALLOWED = "POLICY_SKIP_NOT_ALLOWED"
    POLICY_DEFAULT_NOT_ALLOWED = "POLICY_DEFAULT_NOT_ALLOWED"
    CLARIFICATION_LIMIT_EXCEEDED = "CLARIFICATION_LIMIT_EXCEEDED"
    SKIP_NOT_EXPLICIT = "SKIP_NOT_EXPLICIT"
    CLARIFICATION_CONFLICT = "CLARIFICATION_CONFLICT"
    INVALID_NEXT_ACTION = "INVALID_NEXT_ACTION"
    INVALID_ASSISTANT_REPLY = "INVALID_ASSISTANT_REPLY"


@dataclass(frozen=True, slots=True)
class TurnResponseViolation:
    """
    Описывает одно контекстное нарушение с адресом поля ответа.
    """

    code: TurnResponseViolationCode
    path: str
    message: str


class InvalidTurnResponseError(ValueError):
    """
    Сообщает обо всех найденных нарушениях предложения агента.
    """

    def __init__(self, violations: tuple[TurnResponseViolation, ...]) -> None:
        """
        Сохраняет нарушения и формирует читаемое сообщение исключения.
        """
        self.violations = violations
        details = "; ".join(f"{violation.path}: {violation.message}" for violation in violations)
        super().__init__(f"Ответ агента не прошёл контекстную проверку: {details}")


def collect_turn_response_violations(
    request: TurnRequest,
    response: TurnSuccessResponse,
) -> tuple[TurnResponseViolation, ...]:
    """
    Собирает все детерминированно проверяемые нарушения ответа агента.
    """
    violations: list[TurnResponseViolation] = []
    _validate_response_context(request, response, violations)

    questions_by_code = {
        question.question_code: question for question in request.questionnaire.questions
    }
    states_by_code = {state.question_code: state for state in request.state.questions}
    messages_by_id = {
        message.message_id: message
        for message in (*request.dialog_history, request.current_message)
    }
    projected_statuses = {
        question_code: state.status for question_code, state in states_by_code.items()
    }

    answer_correction = request.answer_correction
    nonadditive_replacements = tuple(
        change
        for change in response.changes
        if isinstance(change, SaveAnswerChange)
        and change.replace_existing
        and not (
            answer_correction is not None
            and answer_correction.mode == AnswerCorrectionMode.ADDITIVE
            and answer_correction.question_code == change.question_code
        )
    )
    if len(nonadditive_replacements) > 1:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.REPLACEMENT_NOT_AUTHORIZED,
                path="changes",
                message="за один ход допустима только одна замена сохранённого ответа",
            )
        )

    for change_index, change in enumerate(response.changes):
        change_path = f"changes[{change_index}]"
        question = questions_by_code.get(change.question_code)
        question_state = states_by_code.get(change.question_code)
        if question is None or question_state is None:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.UNKNOWN_QUESTION,
                    path=f"{change_path}.questionCode",
                    message="изменение ссылается на вопрос, которого нет в анкете",
                )
            )
            continue

        transition_allowed = _validate_state_transition(
            request,
            change,
            question_state,
            change_path,
            violations,
        )
        _validate_change_content(
            request,
            change,
            question,
            question_state,
            messages_by_id,
            change_path,
            violations,
        )
        if transition_allowed:
            projected_statuses[change.question_code] = _projected_status(
                change,
                previous_status=question_state.status,
                is_active=(
                    request.active_question is not None
                    and request.active_question.question_code == change.question_code
                ),
                max_selections=question.max_selections,
                prior_option_codes=(
                    ()
                    if question_state.answer is None
                    else question_state.answer.option_codes
                ),
            )

    clarification_codes = tuple(
        question_code
        for question_code, status in projected_statuses.items()
        if status == QuestionStatus.NEEDS_CLARIFICATION
        or (
            status == QuestionStatus.PARTIALLY_ANSWERED
            and (
                any(
                    isinstance(change, RequestClarificationChange)
                    and change.question_code == question_code
                    for change in response.changes
                )
                or (
                    request.state.pending_clarification is not None
                    and request.state.pending_clarification.question_code == question_code
                    and not any(
                        change.question_code == question_code for change in response.changes
                    )
                )
            )
        )
    )
    if len(clarification_codes) > 1:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.CLARIFICATION_CONFLICT,
                path="changes",
                message=(
                    "после применения патча уточнение ожидается одновременно "
                    "для нескольких вопросов"
                ),
            )
        )

    _validate_next_action(
        request,
        response,
        questions_by_code,
        projected_statuses,
        clarification_codes,
        (request.active_question.question_code if request.active_question is not None else None),
        violations,
    )
    _validate_assistant_reply(
        request,
        response,
        questions_by_code,
        projected_statuses,
        violations,
    )
    return tuple(violations)


def validate_turn_response(
    request: TurnRequest,
    response: TurnSuccessResponse,
) -> None:
    """
    Принимает корректный ответ или поднимает ошибку со всеми нарушениями.
    """
    violations = collect_turn_response_violations(request, response)
    if violations:
        raise InvalidTurnResponseError(violations)


def _validate_response_context(
    request: TurnRequest,
    response: TurnSuccessResponse,
    violations: list[TurnResponseViolation],
) -> None:
    """
    Проверяет, что ответ относится ровно к переданному запросу и состоянию.
    """
    expected_values = (
        ("schemaVersion", response.schema_version, request.schema_version),
        ("correlationId", response.correlation_id, request.correlation_id),
        ("sessionId", response.session_id, request.session_id),
        (
            "sourceMessageId",
            response.source_message_id,
            request.current_message.message_id,
        ),
        ("baseStateVersion", response.base_state_version, request.state.state_version),
        (
            "questionnaireCode",
            response.questionnaire_code,
            request.questionnaire.questionnaire_code,
        ),
        (
            "questionnaireVersion",
            response.questionnaire_version,
            request.questionnaire.questionnaire_version,
        ),
    )
    for path, actual_value, expected_value in expected_values:
        if actual_value != expected_value:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.CONTEXT_MISMATCH,
                    path=path,
                    message="значение не совпадает с исходным запросом",
                )
            )


def _validate_state_transition(
    request: TurnRequest,
    change: StateChange,
    question_state: QuestionState,
    change_path: str,
    violations: list[TurnResponseViolation],
) -> bool:
    """
    Проверяет допустимость операции для исходного статуса вопроса.
    """
    status = question_state.status
    operation_allowed = False
    target_authorized = True
    has_specific_transition_violation = False
    message = "операция недопустима для текущего статуса вопроса"
    active_question = request.active_question
    target_is_active = (
        active_question is not None and active_question.question_code == change.question_code
    )
    bulk_answer_scope = request.answer_scope == TurnAnswerScope.ALL_UNANSWERED

    if isinstance(change, SaveAnswerChange):
        if (
            status in {QuestionStatus.ANSWERED, QuestionStatus.PARTIALLY_ANSWERED}
            and change.replace_existing
        ):
            answer_correction = request.answer_correction
            is_replaceable_answer = (
                status == QuestionStatus.ANSWERED
                and not bulk_answer_scope
                and change.question_code in request.replaceable_question_codes
            )
            if not is_replaceable_answer and (
                answer_correction is None
                or answer_correction.question_code != change.question_code
                or (
                    status == QuestionStatus.PARTIALLY_ANSWERED
                    and answer_correction.mode != AnswerCorrectionMode.ADDITIVE
                )
            ):
                violations.append(
                    TurnResponseViolation(
                        code=TurnResponseViolationCode.REPLACEMENT_NOT_AUTHORIZED,
                        path=f"{change_path}.replaceExisting",
                        message="backend не разрешил замену ответа на этот вопрос",
                    )
                )
                return False
            operation_allowed = True
        elif status == QuestionStatus.ANSWERED:
            if not change.replace_existing:
                message = "замена сохранённого ответа требует replaceExisting=true"
        elif status == QuestionStatus.PARTIALLY_ANSWERED:
            operation_allowed = not bulk_answer_scope and target_is_active
            if not target_is_active:
                target_authorized = False
                _append_inactive_question_violation(change_path, violations)
        elif change.replace_existing:
            message = "replaceExisting=true допустим только для сохранённого ответа"
        else:
            if bulk_answer_scope:
                operation_allowed = status == QuestionStatus.UNKNOWN
                if not operation_allowed:
                    message = "массовое извлечение разрешено только для исходно UNKNOWN-вопроса"
            else:
                operation_allowed = status == QuestionStatus.UNKNOWN or (
                    target_is_active and status == QuestionStatus.NEEDS_CLARIFICATION
                )
                if active_question is None or (
                    not target_is_active and status != QuestionStatus.UNKNOWN
                ):
                    target_authorized = False
                    _append_inactive_question_violation(change_path, violations)
    elif isinstance(change, ReviseMultiChoiceChange):
        correction = request.answer_correction
        operation_allowed = (
            not bulk_answer_scope
            and not target_is_active
            and correction is not None
            and correction.question_code == change.question_code
            and correction.mode == AnswerCorrectionMode.ADDITIVE
            and status in {QuestionStatus.ANSWERED, QuestionStatus.PARTIALLY_ANSWERED}
            and question_state.answer is not None
        )
        if not operation_allowed:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.REPLACEMENT_NOT_AUTHORIZED,
                    path=change_path,
                    message="backend не разрешил пересмотр сохранённого выбора",
                )
            )
            has_specific_transition_violation = True
    elif isinstance(change, ConfirmAnswerChange):
        operation_allowed = (
            not bulk_answer_scope
            and target_is_active
            and change.question_code == "E1"
            and status == QuestionStatus.PARTIALLY_ANSWERED
        )
        if not target_is_active:
            target_authorized = False
            _append_inactive_question_violation(change_path, violations)
    elif isinstance(change, ApplyDefaultAnswerChange):
        operation_allowed = (
            not bulk_answer_scope
            and target_is_active
            and status == QuestionStatus.NEEDS_CLARIFICATION
        )
        if not target_is_active:
            target_authorized = False
            _append_inactive_question_violation(change_path, violations)
    elif isinstance(change, RequestClarificationChange):
        if bulk_answer_scope:
            operation_allowed = status == QuestionStatus.UNKNOWN
            if not operation_allowed:
                message = "массовое уточнение разрешено только для исходно UNKNOWN-вопроса"
        else:
            operation_allowed = status in {
                QuestionStatus.UNKNOWN,
                QuestionStatus.NEEDS_CLARIFICATION,
                QuestionStatus.PARTIALLY_ANSWERED,
            }
            if not target_is_active:
                target_authorized = False
                _append_inactive_question_violation(change_path, violations)
            if question_state.clarification_attempts >= 2:
                operation_allowed = False
                violations.append(
                    TurnResponseViolation(
                        code=TurnResponseViolationCode.CLARIFICATION_LIMIT_EXCEEDED,
                        path=change_path,
                        message="после двух уточнений нельзя задавать третье",
                    )
                )
                has_specific_transition_violation = True
    elif isinstance(change, SkipQuestionChange):
        if bulk_answer_scope:
            message = "answerScope=ALL_UNANSWERED запрещает пропуск вопросов"
        else:
            operation_allowed = status in {
                QuestionStatus.UNKNOWN,
                QuestionStatus.NEEDS_CLARIFICATION,
            }
            if not target_is_active:
                target_authorized = False
                _append_inactive_question_violation(change_path, violations)

    if not operation_allowed and not has_specific_transition_violation:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.INVALID_STATE_TRANSITION,
                path=change_path,
                message=message,
            )
        )
    return operation_allowed and target_authorized


def _append_inactive_question_violation(
    change_path: str,
    violations: list[TurnResponseViolation],
) -> None:
    """
    Добавляет отказ для изменения вопроса вне активного контекста.
    """
    violations.append(
        TurnResponseViolation(
            code=TurnResponseViolationCode.QUESTION_NOT_ACTIVE,
            path=f"{change_path}.questionCode",
            message="обычное изменение допустимо только для activeQuestion",
        )
    )


def _validate_change_content(
    request: TurnRequest,
    change: StateChange,
    question: Question,
    question_state: QuestionState,
    messages_by_id: dict[str, DialogMessage],
    change_path: str,
    violations: list[TurnResponseViolation],
) -> None:
    """
    Проверяет варианты ответа, источники и подтверждающий фрагмент.
    """
    if isinstance(change, SaveAnswerChange):
        correction = request.answer_correction
        additive_prior_codes = (
            question_state.answer.option_codes
            if question_state.answer is not None
            and (
                question_state.status == QuestionStatus.PARTIALLY_ANSWERED
                or (
                    correction is not None
                    and correction.question_code == change.question_code
                    and correction.mode == AnswerCorrectionMode.ADDITIVE
                )
            )
            else None
        )
        _validate_answer_options(
            change,
            question,
            change_path,
            violations,
            additive_prior_codes=additive_prior_codes,
        )
        _validate_sources_and_evidence(
            source_message_ids=change.source_message_ids,
            evidence=change.evidence,
            current_message_id=request.current_message.message_id,
            messages_by_id=messages_by_id,
            expected_source_message_ids=_expected_answer_source_message_ids(
                request,
                change,
                question_state,
            ),
            prior_answer_evidence=(
                question_state.answer.evidence
                if change.replace_existing and question_state.answer is not None
                else None
            ),
            prior_answer_source_ids=(
                question_state.answer.source_message_ids
                if change.replace_existing and question_state.answer is not None
                else ()
            ),
            sources_path=f"{change_path}.sourceMessageIds",
            sources_are_collection=True,
            evidence_path=f"{change_path}.evidence",
            violations=violations,
        )
        if not is_safe_semantic_answer_evidence(change.evidence):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.UNSAFE_ANSWER_EVIDENCE,
                    path=f"{change_path}.evidence",
                    message=(
                        "основание не содержит осмысленного текста, похоже на "
                        "инструкцию агенту или является просьбой пропустить вопрос"
                    ),
                )
            )
    elif isinstance(change, ReviseMultiChoiceChange):
        previous_answer = question_state.answer
        if question.question_type != QuestionType.MULTI_CHOICE or change.question_code != "E1":
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_STATE_TRANSITION,
                    path=change_path,
                    message="пересмотр набора допустим только для E1",
                )
            )
        for option_index, option_code in enumerate(change.option_codes):
            if option_code not in question.allowed_option_codes:
                violations.append(
                    TurnResponseViolation(
                        code=TurnResponseViolationCode.INVALID_OPTION,
                        path=f"{change_path}.optionCodes[{option_index}]",
                        message="вариант исправленного выбора отсутствует в анкете",
                    )
                )
        if len(change.option_codes) > question.max_selections:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_SELECTION_COUNT,
                    path=f"{change_path}.optionCodes",
                    message="исправленный выбор превышает лимит вариантов",
                )
            )
        if previous_answer is not None and set(change.option_codes) == set(
            previous_answer.option_codes
        ):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_STATE_TRANSITION,
                    path=f"{change_path}.optionCodes",
                    message="пересмотр должен менять сохранённый набор",
                )
            )
        prior_source_ids = change.source_message_ids[:-1]
        ordered_user_ids = tuple(
            message.message_id
            for message in request.dialog_history
            if message.role == DialogRole.USER and message.message_id in prior_source_ids
        )
        if prior_source_ids != ordered_user_ids:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_SOURCE_SET,
                    path=f"{change_path}.sourceMessageIds",
                    message="источники исправленного выбора должны идти по истории ребёнка",
                )
            )
        if (
            previous_answer is not None
            and set(change.option_codes) & set(previous_answer.option_codes)
            and not prior_source_ids
        ):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_SOURCE_SET,
                    path=f"{change_path}.sourceMessageIds",
                    message="сохранённый прежний вариант требует прежнего источника",
                )
            )
        _validate_sources_and_evidence(
            source_message_ids=change.source_message_ids,
            evidence=change.evidence,
            current_message_id=request.current_message.message_id,
            messages_by_id=messages_by_id,
            expected_source_message_ids=change.source_message_ids,
            sources_path=f"{change_path}.sourceMessageIds",
            sources_are_collection=True,
            evidence_path=f"{change_path}.evidence",
            violations=violations,
        )
        if not is_safe_semantic_answer_evidence(change.evidence):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.UNSAFE_ANSWER_EVIDENCE,
                    path=f"{change_path}.evidence",
                    message="основание исправления не содержит осмысленного текста",
                )
            )
    elif isinstance(change, ConfirmAnswerChange):
        _validate_sources_and_evidence(
            source_message_ids=(change.source_message_id,),
            evidence=change.evidence,
            current_message_id=request.current_message.message_id,
            messages_by_id=messages_by_id,
            expected_source_message_ids=(request.current_message.message_id,),
            sources_path=f"{change_path}.sourceMessageId",
            sources_are_collection=False,
            evidence_path=f"{change_path}.evidence",
            violations=violations,
        )
    elif isinstance(change, ApplyDefaultAnswerChange):
        zero_option_codes = tuple(
            option.option_code for option in question.options if option.numeric_value == 0
        )
        pending = request.state.pending_clarification
        if (
            question.question_type != QuestionType.SCALE_0_3
            or change.option_codes != zero_option_codes
            or pending is None
            or pending.question_code != change.question_code
            or question_state.clarification_attempts != 2
        ):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.POLICY_DEFAULT_NOT_ALLOWED,
                    path=change_path,
                    message=(
                        "нулевой ответ по правилу допустим только после двух "
                        "уточнений активного вопроса со шкалой 0–3"
                    ),
                )
            )
    elif isinstance(change, SkipQuestionChange):
        _validate_sources_and_evidence(
            source_message_ids=(change.source_message_id,),
            evidence=change.evidence,
            current_message_id=request.current_message.message_id,
            messages_by_id=messages_by_id,
            expected_source_message_ids=(request.current_message.message_id,),
            sources_path=f"{change_path}.sourceMessageId",
            sources_are_collection=False,
            evidence_path=f"{change_path}.evidence",
            violations=violations,
        )
        is_policy_skip = change.reason == PolicySkipReason.NO_ANSWER_AFTER_RETRY
        if is_policy_skip and not _is_allowed_policy_skip(request, change):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.POLICY_SKIP_NOT_ALLOWED,
                    path=f"{change_path}.reason",
                    message=(
                        "программный пропуск допустим только после двух уточнений активного вопроса"
                    ),
                )
            )
        elif change.reason == PolicySkipReason.NO_MATCHING_OPTIONS:
            if not _is_allowed_no_matching_options_skip(
                request,
                change,
                question,
                messages_by_id,
            ):
                violations.append(
                    TurnResponseViolation(
                        code=TurnResponseViolationCode.POLICY_SKIP_NOT_ALLOWED,
                        path=f"{change_path}.reason",
                        message=(
                            "пропуск без подходящего варианта допустим только после "
                            "уточнения списка активного вопроса"
                        ),
                    )
                )
        elif not is_policy_skip and not is_explicit_skip_evidence(change.evidence):
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.SKIP_NOT_EXPLICIT,
                    path=f"{change_path}.evidence",
                    message="основание не содержит явного отказа или просьбы перейти дальше",
                )
            )


def _is_allowed_policy_skip(
    request: TurnRequest,
    change: SkipQuestionChange,
) -> bool:
    """
    Разрешает служебный пропуск только после исчерпания лимита уточнений.
    """
    active_question = request.active_question
    pending = request.state.pending_clarification
    question_state = next(
        (item for item in request.state.questions if item.question_code == change.question_code),
        None,
    )
    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == change.question_code
        ),
        None,
    )
    return (
        request.answer_scope == TurnAnswerScope.ACTIVE_QUESTION
        and active_question is not None
        and active_question.question_code == change.question_code
        and pending is not None
        and pending.question_code == change.question_code
        and question_state is not None
        and question_state.status == QuestionStatus.NEEDS_CLARIFICATION
        and question_state.clarification_attempts >= 2
        and question is not None
        and question.question_type != QuestionType.SCALE_0_3
    )


def _is_allowed_no_matching_options_skip(
    request: TurnRequest,
    change: SkipQuestionChange,
    question: Question,
    messages_by_id: dict[str, DialogMessage],
) -> bool:
    """
    Разрешает пропуск после уточнения или объяснения активного выбора.
    """
    active_question = request.active_question
    pending = request.state.pending_clarification
    question_state = next(
        (item for item in request.state.questions if item.question_code == change.question_code),
        None,
    )
    if (
        request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or question.question_type != QuestionType.MULTI_CHOICE
        or active_question is None
        or active_question.question_code != change.question_code
        or question_state is None
    ):
        return False
    active_message = messages_by_id.get(active_question.assistant_message_id)
    if active_message is None or active_message.role != DialogRole.ASSISTANT:
        return False
    if pending is not None:
        return (
            active_question.assistant_message_id == pending.assistant_message_id
            and pending.question_code == change.question_code
            and question_state.status == QuestionStatus.NEEDS_CLARIFICATION
            and question_state.clarification_attempts >= 1
        )
    if (
        question_state.status != QuestionStatus.UNKNOWN
        or active_message.content == question.question_text
    ):
        return False
    active_position = next(
        index
        for index, message in enumerate(request.dialog_history)
        if message.message_id == active_question.assistant_message_id
    )
    return any(
        message.role == DialogRole.ASSISTANT
        and _is_safe_ask_question_reply(message.content, question.question_text)
        and any(
            intervening.role == DialogRole.USER
            for intervening in request.dialog_history[index + 1 : active_position]
        )
        for index, message in enumerate(request.dialog_history[:active_position])
    )


def _expected_answer_source_message_ids(
    request: TurnRequest,
    change: SaveAnswerChange,
    question_state: QuestionState,
) -> tuple[str, ...]:
    """
    Возвращает точную цепочку источников, включая прежний ответ при его дополнении.
    """
    pending = request.state.pending_clarification
    context_ids = (
        pending.context_message_ids
        if pending is not None and pending.question_code == change.question_code
        else ()
    )
    current_chain = tuple(dict.fromkeys((*context_ids, request.current_message.message_id)))
    correction = request.answer_correction
    may_reuse_prior_sources = (
        change.replace_existing
        and question_state.status == QuestionStatus.ANSWERED
        and question_state.answer is not None
        and (
            change.question_code in request.replaceable_question_codes
            or (
                correction is not None
                and correction.question_code == change.question_code
                and correction.mode == AnswerCorrectionMode.REPLACE
            )
        )
    )
    if may_reuse_prior_sources:
        combined_chain = tuple(
            dict.fromkeys((*question_state.answer.source_message_ids, *current_chain))
        )
        if change.source_message_ids == combined_chain:
            return combined_chain
    return current_chain


def _validate_answer_options(
    change: SaveAnswerChange,
    question: Question,
    change_path: str,
    violations: list[TurnResponseViolation],
    *,
    additive_prior_codes: tuple[str, ...] | None = None,
) -> None:
    """
    Проверяет принадлежность вариантов и размер итогового набора при дополнении.
    """
    allowed_codes = set(question.allowed_option_codes)
    for option_index, option_code in enumerate(change.option_codes):
        if option_code not in allowed_codes:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_OPTION,
                    path=f"{change_path}.optionCodes[{option_index}]",
                    message="вариант ответа не принадлежит указанному вопросу",
                )
            )

    selection_count = len(change.option_codes)
    if additive_prior_codes is not None:
        repeated_codes = set(change.option_codes) & set(additive_prior_codes)
        if repeated_codes:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.ADDITIVE_OPTION_CONFLICT,
                    path=f"{change_path}.optionCodes",
                    message="дополнение может содержать только ещё не сохранённые варианты",
                )
            )
        selection_count += len(additive_prior_codes)
    if not question.min_selections <= selection_count <= question.max_selections:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.INVALID_SELECTION_COUNT,
                path=f"{change_path}.optionCodes",
                message=(
                    "число выбранных вариантов не входит в допустимый "
                    f"диапазон {question.min_selections}..{question.max_selections}"
                ),
            )
        )


def _validate_sources_and_evidence(
    *,
    source_message_ids: tuple[str, ...],
    evidence: str,
    current_message_id: str,
    messages_by_id: dict[str, DialogMessage],
    expected_source_message_ids: tuple[str, ...],
    sources_path: str,
    sources_are_collection: bool,
    evidence_path: str,
    violations: list[TurnResponseViolation],
    prior_answer_evidence: str | None = None,
    prior_answer_source_ids: tuple[str, ...] = (),
) -> None:
    """
    Проверяет пользовательские источники и дословность подтверждения.
    """
    if source_message_ids != expected_source_message_ids:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.INVALID_SOURCE_SET,
                path=sources_path,
                message="источники не совпадают с полной цепочкой текущего ответа",
            )
        )

    referenced_user_messages: list[DialogMessage] = []
    for source_index, message_id in enumerate(source_message_ids):
        source_path = f"{sources_path}[{source_index}]" if sources_are_collection else sources_path
        message = messages_by_id.get(message_id)
        if message is None:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.UNKNOWN_SOURCE_MESSAGE,
                    path=source_path,
                    message="источник отсутствует в истории и текущей реплике",
                )
            )
        elif message.role != DialogRole.USER:
            violations.append(
                TurnResponseViolation(
                    code=TurnResponseViolationCode.INVALID_SOURCE_ROLE,
                    path=source_path,
                    message="источником изменения может быть только реплика пользователя",
                )
            )
        else:
            referenced_user_messages.append(message)

    if current_message_id not in source_message_ids:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.CURRENT_MESSAGE_NOT_REFERENCED,
                path=sources_path,
                message="изменение должно ссылаться на текущую реплику пользователя",
            )
        )

    evidence_is_valid = (
        _valid_chained_evidence(
            evidence,
            expected_source_message_ids,
            messages_by_id,
            prior_answer_evidence=prior_answer_evidence,
            prior_answer_source_ids=prior_answer_source_ids,
        )
        if len(expected_source_message_ids) > 1
        else any(
            _normalize_evidence_text(evidence) in _normalize_evidence_text(message.content)
            for message in referenced_user_messages
        )
    )
    if not evidence_is_valid:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.EVIDENCE_NOT_FOUND,
                path=evidence_path,
                message="подтверждающий фрагмент не найден в указанных источниках",
            )
        )


def _valid_chained_evidence(
    evidence: str,
    expected_source_message_ids: tuple[str, ...],
    messages_by_id: dict[str, DialogMessage],
    *,
    prior_answer_evidence: str | None = None,
    prior_answer_source_ids: tuple[str, ...] = (),
) -> bool:
    """
    Проверяет текущую цитату либо составное основание из разрешённых источников.
    """
    source_messages = tuple(
        messages_by_id.get(message_id) for message_id in expected_source_message_ids
    )
    if any(message is None or message.role != DialogRole.USER for message in source_messages):
        return False

    previous_messages = source_messages[:-1]
    current_message = source_messages[-1]
    assert current_message is not None
    if _normalize_evidence_text(evidence) in _normalize_evidence_text(current_message.content):
        return True
    if (
        prior_answer_evidence is not None
        and prior_answer_source_ids
        and expected_source_message_ids[: len(prior_answer_source_ids)]
        == prior_answer_source_ids
        and _prior_saved_evidence_is_grounded(
            prior_answer_evidence,
            prior_answer_source_ids,
            messages_by_id,
        )
    ):
        prefix = f"{prior_answer_evidence}; "
        if evidence.startswith(prefix):
            current_fragment = evidence[len(prefix) :]
            if current_fragment and _normalize_evidence_text(
                current_fragment
            ) in _normalize_evidence_text(current_message.content):
                return True
    prefix = "; ".join(message.content for message in previous_messages if message is not None)
    prefix_with_separator = f"{prefix}; "
    if not evidence.startswith(prefix_with_separator):
        return False
    current_fragment = evidence[len(prefix_with_separator) :]
    return bool(current_fragment) and _normalize_evidence_text(
        current_fragment
    ) in _normalize_evidence_text(current_message.content)


def _prior_saved_evidence_is_grounded(
    evidence: str,
    source_message_ids: tuple[str, ...],
    messages_by_id: dict[str, DialogMessage],
) -> bool:
    """
    Подтверждает старую цитату только её исходными детскими репликами.
    """
    if len(source_message_ids) > 1:
        return _valid_chained_evidence(evidence, source_message_ids, messages_by_id)
    message = messages_by_id.get(source_message_ids[0])
    return (
        message is not None
        and message.role == DialogRole.USER
        and _normalize_evidence_text(evidence)
        in _normalize_evidence_text(message.content)
    )


def _normalize_evidence_text(value: str) -> str:
    """
    Нормализует Unicode, регистр и пробелы без нечёткого сопоставления.
    """
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _validate_assistant_reply(
    request: TurnRequest,
    response: TurnSuccessResponse,
    questions_by_code: dict[str, Question],
    projected_statuses: dict[str, QuestionStatus],
    violations: list[TurnResponseViolation],
) -> None:
    """
    Связывает пользовательскую реплику с проверенным следующим действием.
    """
    reply = response.assistant_reply
    if any(
        is_forbidden_dialog_text_character(character) for character in reply
    ) or has_invalid_dialog_joiner_context(reply):
        _append_invalid_assistant_reply(
            violations,
            "реплика содержит запрещённый управляющий Unicode-символ",
        )
        return

    next_action = response.next_action
    question_code = next_action.question_code
    companion_code = next_action.companion_question_code
    if next_action.action == NextActionType.REASK_QUESTION:
        if question_code is None or question_code not in questions_by_code:
            return
        is_partial = projected_statuses[question_code] == QuestionStatus.PARTIALLY_ANSWERED
        if is_partial:
            expected_reask = partial_multi_choice_prompt(
                questions_by_code[question_code],
                _projected_option_codes(request, response, question_code),
            )
        else:
            expected_reask = canonical_turn_reply(
                request.questionnaire,
                NextActionType.REASK_QUESTION,
                question_code,
                pending_clarification=request.state.pending_clarification is not None,
                corrected_answer=any(
                    isinstance(change, SaveAnswerChange)
                    and change.replace_existing
                    and change.question_code in request.replaceable_question_codes
                    for change in response.changes
                ),
                recorded_other_answer=any(
                    isinstance(change, SaveAnswerChange)
                    and change.question_code != question_code
                    for change in response.changes
                ),
            )
        question = questions_by_code[question_code]
        natural_explanation = (
            (not is_partial or question_code != "E1" or _is_open_partial_choice_reply(reply))
            and _valid_explanatory_reask_reply(request, question, reply)
        )
        if reply != expected_reask and not natural_explanation:
            _append_invalid_assistant_reply(
                violations,
                "повтор должен совпадать с канонической репликой либо безопасно объяснять активный вопрос",
            )
        return
    if next_action.action in {
        NextActionType.ASK_QUESTION,
        NextActionType.CLARIFY,
    }:
        if question_code is None:
            return
        question = questions_by_code.get(question_code)
        if question is None:
            return

        clarification_change = next(
            (
                change
                for change in response.changes
                if isinstance(change, RequestClarificationChange)
            ),
            None,
        )
        expected_text = question.question_text
        is_partial = projected_statuses[question_code] == QuestionStatus.PARTIALLY_ANSWERED
        if next_action.action == NextActionType.ASK_QUESTION and is_partial:
            expected_text = partial_multi_choice_prompt(
                question,
                _projected_option_codes(request, response, question_code),
            )
        elif next_action.action == NextActionType.CLARIFY:
            if clarification_change is None:
                return
            expected_text = clarification_change.clarification_text
        if next_action.action == NextActionType.CLARIFY:
            reply_matches = reply == expected_text
        elif companion_code is not None:
            composite = next(
                (
                    item
                    for item in request.questionnaire.composite_questions
                    if item.question_codes == (question_code, companion_code)
                ),
                None,
            )
            reply_matches = composite is not None and (
                reply == composite.question_text
                or _is_safe_natural_ask_question_reply(request, reply)
            )
        elif is_partial:
            reply_matches = reply == expected_text or (
                question_code == "E1"
                and _is_open_partial_choice_reply(reply)
                and _is_safe_natural_ask_question_reply(request, reply)
            )
        else:
            reply_matches = (
                _is_safe_ask_question_reply(reply, expected_text)
                if reply.endswith(expected_text)
                else _is_safe_natural_ask_question_reply(request, reply)
            )
        if not reply_matches:
            _append_invalid_assistant_reply(
                violations,
                "реплика не совпадает с выбранным вопросом или уточнением",
            )
            return

        if next_action.action == NextActionType.CLARIFY and not (
            is_partial
            and clarification_change is not None
            and reply
            == partial_multi_choice_followup(
                question,
                _projected_option_codes(request, response, question_code),
                next(
                    state.clarification_attempts
                    for state in request.state.questions
                    if state.question_code == question_code
                )
                + 1,
            )
            or _valid_clarification_reply(request, question, reply)
        ):
            _append_invalid_assistant_reply(
                violations,
                "уточнение небезопасно, повторяется или не связано с вопросом",
            )
            return

        for other_code, other_question in questions_by_code.items():
            if other_code == question_code:
                continue
            normalized_other_prompts = (
                _normalize_evidence_text(prompt_text)
                for prompt_text in _question_prompts(other_question)
            )
            if any(
                prompt_text not in _normalize_evidence_text(expected_text)
                and prompt_text in _normalize_evidence_text(reply)
                for prompt_text in normalized_other_prompts
            ):
                _append_invalid_assistant_reply(
                    violations,
                    "реплика одновременно содержит текст другого вопроса анкеты",
                )
                return
        return

    normalized_reply = _normalize_evidence_text(reply)
    contains_question_prompt = any(
        _normalize_evidence_text(prompt_text) in normalized_reply
        for question in questions_by_code.values()
        for prompt_text in _question_prompts(question)
    )
    if contains_question_prompt or any(_is_question_mark(character) for character in reply):
        _append_invalid_assistant_reply(
            violations,
            "реплика завершения не должна задавать новый вопрос",
        )


def _projected_option_codes(
    request: TurnRequest,
    response: TurnSuccessResponse,
    question_code: str,
) -> tuple[str, ...]:
    """
    Возвращает сохранённые и добавленные варианты частичного ответа.
    """
    state = next(item for item in request.state.questions if item.question_code == question_code)
    prior_codes = state.answer.option_codes if state.answer is not None else ()
    change = next(
        (item for item in response.changes if item.question_code == question_code),
        None,
    )
    if isinstance(change, SaveAnswerChange):
        if state.status == QuestionStatus.PARTIALLY_ANSWERED:
            return tuple(dict.fromkeys((*prior_codes, *change.option_codes)))
        return change.option_codes
    if isinstance(change, ReviseMultiChoiceChange):
        return change.option_codes
    return prior_codes


def _is_safe_ask_question_reply(reply: str, question_text: str) -> bool:
    """
    Разрешает точный вопрос с необязательным коротким безопасным переходом перед ним.
    """
    if not reply.endswith(question_text):
        return False

    prefix = reply[: -len(question_text)]
    if not prefix:
        return True
    if not prefix.endswith(" ") or prefix.endswith("  "):
        return False

    transition = prefix[:-1]
    if transition != transition.strip() or not transition:
        return False
    if len(transition) > _MAX_TRANSITION_PREFIX_LENGTH:
        return False
    if any(_is_question_mark(character) for character in transition):
        return False
    if repeats_official_question(transition, question_text):
        return False
    first_word_match = _SHORT_ANSWER_WORD_PATTERN.search(transition)
    if (
        first_word_match is not None
        and first_word_match.group().casefold() in _MECHANICAL_TRANSITION_PREFIXES
    ):
        return False

    lowered_transition = transition.casefold()
    if any(marker in lowered_transition for marker in _UNSAFE_TRANSITION_MARKERS):
        return False
    if _EMAIL_PATTERN.search(transition) or _PHONE_NUMBER_PATTERN.search(transition):
        return False
    return _UNSAFE_TRANSITION_WORD_PATTERN.search(transition) is None


def _is_safe_natural_ask_question_reply(request: TurnRequest, reply: str) -> bool:
    """
    Проверяет форму и безопасность цельного вопроса без оценки его смысла по словам.
    """
    if (
        reply != reply.strip()
        or not 1 <= len(reply) <= _MAX_NATURAL_QUESTION_LENGTH
        or any(character.isspace() and character != " " for character in reply)
        or "  " in reply
    ):
        return False
    if sum(_is_question_mark(character) for character in reply) != 1:
        return False
    question_ending = reply.rstrip(" 🙂😊")
    if not question_ending or not _is_question_mark(question_ending[-1]):
        return False
    if any(marker in reply.casefold() for marker in _UNSAFE_TRANSITION_MARKERS):
        return False
    if (
        _EMAIL_PATTERN.search(reply)
        or _PHONE_NUMBER_PATTERN.search(reply)
        or _UNSAFE_TRANSITION_WORD_PATTERN.search(reply)
        or _UNSAFE_EXPLANATION_WORD_PATTERN.search(reply)
        or _EXPLANATION_OPTION_REFERENCE_PATTERN.search(reply)
    ):
        return False
    symbols = tuple(
        character for character in reply if unicodedata.category(character).startswith("S")
    )
    if len(symbols) > 1 or any(
        character not in _FRIENDLY_EXPLANATION_SYMBOLS for character in symbols
    ):
        return False
    previous_assistant_text = next(
        (
            message.content
            for message in reversed(request.dialog_history)
            if message.role == DialogRole.ASSISTANT
        ),
        None,
    )
    if previous_assistant_text is not None:
        previous_words = tuple(
            _SHORT_ANSWER_WORD_PATTERN.findall(
                unicodedata.normalize("NFKC", previous_assistant_text).casefold()
            )
        )
        reply_words = tuple(
            _SHORT_ANSWER_WORD_PATTERN.findall(
                unicodedata.normalize("NFKC", reply).casefold()
            )
        )
        if reply_words == previous_words:
            return False
    return True


def _is_open_partial_choice_reply(reply: str) -> bool:
    """
    Не даёт сузить вопрос о всех оставшихся занятиях до одного примера.
    """
    return bool(
        _OPEN_ADDITIONAL_CHOICE_SUBJECT.search(reply)
        and _OPEN_ADDITIONAL_CHOICE_SCOPE.search(reply)
    )


def _question_prompts(question: Question) -> tuple[str, ...]:
    """
    Возвращает канонические тексты вопроса и отдельного уточнения.
    """
    if question.clarification_text is None:
        return (question.question_text,)
    return (question.question_text, question.clarification_text)


def _valid_clarification_reply(
    request: TurnRequest,
    question: Question,
    reply: str,
) -> bool:
    """
    Проверяет форму, явную безопасность и отсутствие повтора уточнения.
    """
    previous_assistant_text = (
        request.dialog_history[-1].content
        if request.dialog_history and request.dialog_history[-1].role == DialogRole.ASSISTANT
        else None
    )
    return is_safe_contextual_clarification(
        question,
        reply,
        previous_assistant_text=previous_assistant_text,
        context_texts=relevant_clarification_context_texts(
            request,
            question.question_code,
        ),
    )


def _valid_explanatory_reask_reply(
    request: TurnRequest,
    question: Question,
    reply: str,
) -> bool:
    """
    Проверяет пояснение и точное совпадение итогового вопроса с активным.
    """
    if reply != reply.strip() or not 1 <= len(reply) <= 400:
        return False
    if any(marker in reply.casefold() for marker in _UNSAFE_TRANSITION_MARKERS):
        return False
    if (
        _EMAIL_PATTERN.search(reply)
        or _PHONE_NUMBER_PATTERN.search(reply)
        or _UNSAFE_TRANSITION_WORD_PATTERN.search(reply)
        or _UNSAFE_EXPLANATION_WORD_PATTERN.search(reply)
        or _EXPLANATION_OPTION_REFERENCE_PATTERN.search(reply)
    ):
        return False
    symbols = tuple(
        character
        for character in reply
        if unicodedata.category(character).startswith("S")
    )
    if len(symbols) > 1 or any(
        character not in _FRIENDLY_EXPLANATION_SYMBOLS for character in symbols
    ):
        return False
    if sum(_is_question_mark(character) for character in reply) != 1:
        return False
    question_ending = reply.rstrip(" 🙂😊")
    if not question_ending or not _is_question_mark(question_ending[-1]):
        return False
    pending = request.state.pending_clarification
    trusted_question = (
        question.followup_clarification_text
        if pending is not None
        and pending.question_code == question.question_code
        and question.followup_clarification_text is not None
        else question.clarification_text or question.question_text
    )
    if not reply.endswith(trusted_question):
        return False
    introduction = reply[: -len(trusted_question)].rstrip()
    return not introduction or introduction[-1] in ".!"


def _is_question_mark(character: str) -> bool:
    """
    Распознаёт вопросительные знаки и интерробанги по имени Unicode.
    """
    unicode_name = unicodedata.name(character, "")
    return "INTERROBANG" in unicode_name or ("QUESTION" in unicode_name and "MARK" in unicode_name)


def _append_invalid_assistant_reply(
    violations: list[TurnResponseViolation],
    message: str,
) -> None:
    """
    Добавляет безопасное нарушение текста ответа ассистента.
    """
    violations.append(
        TurnResponseViolation(
            code=TurnResponseViolationCode.INVALID_ASSISTANT_REPLY,
            path="assistantReply",
            message=message,
        )
    )


def _projected_status(
    change: StateChange,
    *,
    previous_status: QuestionStatus,
    is_active: bool,
    max_selections: int,
    prior_option_codes: tuple[str, ...],
) -> QuestionStatus:
    """
    Возвращает статус вопроса после допустимой операции без мутации снимка.
    """
    if isinstance(change, SaveAnswerChange):
        if (
            change.question_code == "E1"
            and previous_status in {
                QuestionStatus.PARTIALLY_ANSWERED,
                QuestionStatus.UNKNOWN,
            }
            and not is_active
            and len(set((*prior_option_codes, *change.option_codes))) < max_selections
        ):
            return QuestionStatus.PARTIALLY_ANSWERED
        return QuestionStatus.ANSWERED
    if isinstance(change, ReviseMultiChoiceChange):
        if not change.option_codes:
            return QuestionStatus.UNKNOWN
        if previous_status == QuestionStatus.PARTIALLY_ANSWERED and not is_active:
            return QuestionStatus.PARTIALLY_ANSWERED
        return QuestionStatus.ANSWERED
    if isinstance(change, (ConfirmAnswerChange, ApplyDefaultAnswerChange)):
        return QuestionStatus.ANSWERED
    if isinstance(change, RequestClarificationChange):
        if previous_status == QuestionStatus.PARTIALLY_ANSWERED:
            return QuestionStatus.PARTIALLY_ANSWERED
        return QuestionStatus.NEEDS_CLARIFICATION
    return QuestionStatus.SKIPPED


def _validate_next_action(
    request: TurnRequest,
    response: TurnSuccessResponse,
    questions_by_code: dict[str, Question],
    projected_statuses: dict[str, QuestionStatus],
    clarification_codes: tuple[str, ...],
    active_question_code: str | None,
    violations: list[TurnResponseViolation],
) -> None:
    """
    Проверяет следующее действие по спроецированным статусам вопросов.
    """
    next_action = response.next_action
    question_code = next_action.question_code
    companion_code = next_action.companion_question_code
    if question_code is not None and question_code not in questions_by_code:
        violations.append(
            TurnResponseViolation(
                code=TurnResponseViolationCode.UNKNOWN_QUESTION,
                path="nextAction.questionCode",
                message="следующее действие ссылается на вопрос, которого нет в анкете",
            )
        )
        return

    terminal_statuses = {
        QuestionStatus.ANSWERED,
        QuestionStatus.SKIPPED,
        QuestionStatus.NOT_APPLICABLE,
    }
    all_terminal = all(status in terminal_statuses for status in projected_statuses.values())
    has_skips = QuestionStatus.SKIPPED in projected_statuses.values()
    answered_companion_only = (
        request.active_question is not None
        and request.active_question.companion_question_code is not None
        and question_code == active_question_code
        and any(
            isinstance(change, SaveAnswerChange)
            and change.question_code == request.active_question.companion_question_code
            for change in response.changes
        )
    )

    if next_action.action == NextActionType.ASK_QUESTION:
        if companion_code is not None:
            composite = next(
                (
                    item
                    for item in request.questionnaire.composite_questions
                    if item.question_codes == (question_code, companion_code)
                ),
                None,
            )
            if composite is None or projected_statuses.get(companion_code) != QuestionStatus.UNKNOWN:
                _append_invalid_next_action(
                    violations,
                    "составной вопрос требует объявленную пару ещё открытых вопросов",
                )
        if question_code is None or projected_statuses[question_code] not in {
            QuestionStatus.UNKNOWN,
            QuestionStatus.PARTIALLY_ANSWERED,
        }:
            _append_invalid_next_action(
                violations,
                "ASK_QUESTION должен указывать открытый вопрос после патча",
            )
        elif question_code == "E1" and any(
            code != question_code and status == QuestionStatus.UNKNOWN
            for code, status in projected_statuses.items()
        ):
            _append_invalid_next_action(
                violations,
                "E1 задаётся после остальных открытых вопросов",
            )
        elif projected_statuses[question_code] == QuestionStatus.PARTIALLY_ANSWERED and question_code != "E1":
            _append_invalid_next_action(
                violations,
                "частичный ответ допустим только для E1",
            )
        elif clarification_codes:
            _append_invalid_next_action(
                violations,
                "нельзя задавать новый вопрос, пока ожидается уточнение",
            )
        elif (
            active_question_code is not None
            and projected_statuses[active_question_code] == QuestionStatus.UNKNOWN
            and not answered_companion_only
        ):
            _append_invalid_next_action(
                violations,
                "activeQuestion требует изменения состояния перед следующим вопросом",
            )
        elif (
            question_code == active_question_code
            and projected_statuses[question_code] == QuestionStatus.PARTIALLY_ANSWERED
        ):
            _append_invalid_next_action(
                violations,
                "частично заполненный активный E1 требует ответа или уточнения",
            )
    elif next_action.action == NextActionType.REASK_QUESTION:
        original_statuses = {
            state.question_code: state.status for state in request.state.questions
        }
        answer_correction = request.answer_correction
        replaceable_changes = tuple(
            change
            for change in response.changes
            if isinstance(change, SaveAnswerChange)
            and change.replace_existing
            and change.question_code in request.replaceable_question_codes
        )
        only_permitted_non_active_answers = (
            len(replaceable_changes) <= 1
            and all(
                (
                    isinstance(change, ReviseMultiChoiceChange)
                    and change.question_code != active_question_code
                    and answer_correction is not None
                    and answer_correction.mode == AnswerCorrectionMode.ADDITIVE
                    and answer_correction.question_code == change.question_code
                )
                or (
                    isinstance(change, SaveAnswerChange)
                    and change.question_code != active_question_code
                    and (
                        (
                            change.replace_existing
                            and (
                                change.question_code in request.replaceable_question_codes
                                or (
                                    answer_correction is not None
                                    and answer_correction.mode == AnswerCorrectionMode.ADDITIVE
                                    and answer_correction.question_code == change.question_code
                                )
                            )
                        )
                        or (
                            not change.replace_existing
                            and original_statuses.get(change.question_code)
                            == QuestionStatus.UNKNOWN
                        )
                    )
                )
                for change in response.changes
            )
        )
        if (
            request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
            or question_code is None
            or question_code != active_question_code
            or projected_statuses[question_code]
            not in {
                QuestionStatus.UNKNOWN,
                QuestionStatus.NEEDS_CLARIFICATION,
                QuestionStatus.PARTIALLY_ANSWERED,
            }
            or not only_permitted_non_active_answers
        ):
            _append_invalid_next_action(
                violations,
                "REASK_QUESTION требует неизменённого активного вопроса; допустимы только попутные ответы и не более одной авторизованной коррекции",
            )
    elif next_action.action == NextActionType.CLARIFY:
        if (
            question_code is None
            or projected_statuses[question_code]
            not in {
                QuestionStatus.NEEDS_CLARIFICATION,
                QuestionStatus.PARTIALLY_ANSWERED,
            }
            or clarification_codes != (question_code,)
            or (
                projected_statuses[question_code] == QuestionStatus.PARTIALLY_ANSWERED
                and question_code != "E1"
            )
        ):
            _append_invalid_next_action(
                violations,
                "CLARIFY должен указывать единственный вопрос, ожидающий уточнения",
            )
    elif next_action.action == NextActionType.COMPLETE:
        if not all_terminal or has_skips:
            _append_invalid_next_action(
                violations,
                "COMPLETE требует завершения всех вопросов без пропусков",
            )
    elif next_action.action == NextActionType.COMPLETE_WITH_SKIPS:
        if not all_terminal or not has_skips:
            _append_invalid_next_action(
                violations,
                "COMPLETE_WITH_SKIPS требует завершения и хотя бы одного пропуска",
            )


def _append_invalid_next_action(
    violations: list[TurnResponseViolation],
    message: str,
) -> None:
    """
    Добавляет нарушение общего поля следующего действия.
    """
    violations.append(
        TurnResponseViolation(
            code=TurnResponseViolationCode.INVALID_NEXT_ACTION,
            path="nextAction",
            message=message,
        )
    )

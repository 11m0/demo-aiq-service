"""
Проецирует проверенный ML-патч в новый неизменяемый снимок сессии.
"""

from dataclasses import dataclass

from questionnaire_backend.repository import QuestionnaireSession
from questionnaire_contracts import (
    ActiveQuestionContext,
    AnswerCorrectionAuthorization,
    AnswerCorrectionMode,
    AnswerDefaultReason,
    AnswerOrigin,
    ApplyDefaultAnswerChange,
    ConfirmAnswerChange,
    DialogMessage,
    NextActionType,
    PendingClarification,
    QuestionnaireState,
    QuestionnaireStatus,
    QuestionState,
    QuestionStatus,
    RequestClarificationChange,
    ReviseMultiChoiceChange,
    SaveAnswerChange,
    SavedAnswer,
    SkipQuestionChange,
    TurnSuccessResponse,
)


@dataclass(frozen=True, slots=True)
class TurnProjection:
    """
    Содержит все поля, которые нужно атомарно зафиксировать после хода.
    """

    state: QuestionnaireState
    active_question: ActiveQuestionContext | None
    dialog_history: tuple[DialogMessage, ...]


def project_successful_turn(
    session: QuestionnaireSession,
    user_message: DialogMessage,
    assistant_message: DialogMessage,
    response: TurnSuccessResponse,
    *,
    answer_correction: AnswerCorrectionAuthorization | None = None,
    replaceable_question_codes: tuple[str, ...] = (),
) -> TurnProjection:
    """
    Строит новый снимок, не изменяя исходный агрегат при ошибке вычисления.
    """
    states_by_code = {item.question_code: item for item in session.state.questions}
    clarification_reason: str | None = None

    for change in response.changes:
        previous = states_by_code[change.question_code]
        if isinstance(change, SaveAnswerChange):
            question = next(
                item
                for item in session.questionnaire.questions
                if item.question_code == change.question_code
            )
            states_by_code[change.question_code] = _apply_saved_answer(
                previous,
                change,
                answer_correction=answer_correction,
                active_question_code=(
                    None if session.active_question is None else session.active_question.question_code
                ),
                max_selections=question.max_selections,
            )
        elif isinstance(change, ReviseMultiChoiceChange):
            states_by_code[change.question_code] = _apply_revised_multi_choice(
                previous,
                change,
                answer_correction=answer_correction,
                active_question_code=(
                    None if session.active_question is None else session.active_question.question_code
                ),
            )
        elif isinstance(change, ConfirmAnswerChange):
            states_by_code[change.question_code] = _confirm_partial_answer(
                previous,
                active_question_code=(
                    None if session.active_question is None else session.active_question.question_code
                ),
            )
        elif isinstance(change, ApplyDefaultAnswerChange):
            states_by_code[change.question_code] = _apply_default_answer(previous, change)
        elif isinstance(change, RequestClarificationChange):
            clarification_reason = change.reason
            states_by_code[change.question_code] = _apply_clarification(
                previous,
                change,
            )
        elif isinstance(change, SkipQuestionChange):
            states_by_code[change.question_code] = _apply_skip(previous, change)

    next_action = response.next_action
    if next_action.action == NextActionType.ASK_QUESTION:
        questionnaire_status = QuestionnaireStatus.IN_PROGRESS
        active_question = ActiveQuestionContext(
            question_code=next_action.question_code,
            assistant_message_id=assistant_message.message_id,
            companion_question_code=next_action.companion_question_code,
        )
        pending_clarification = None
    elif next_action.action == NextActionType.REASK_QUESTION:
        questionnaire_status = QuestionnaireStatus.IN_PROGRESS
        active_question = ActiveQuestionContext(
            question_code=next_action.question_code,
            assistant_message_id=assistant_message.message_id,
        )
        previous_pending = session.state.pending_clarification
        answered_other_question = any(
            isinstance(change, SaveAnswerChange)
            and change.question_code != next_action.question_code
            for change in response.changes
        )
        pending_clarification = (
            PendingClarification(
                question_code=previous_pending.question_code,
                assistant_message_id=assistant_message.message_id,
                reason=previous_pending.reason,
                context_message_ids=(
                    previous_pending.context_message_ids
                    if answered_other_question
                    else tuple(
                        dict.fromkeys(
                            (*previous_pending.context_message_ids, user_message.message_id)
                        )
                    )
                ),
            )
            if previous_pending is not None
            else None
        )
    elif next_action.action == NextActionType.CLARIFY:
        questionnaire_status = QuestionnaireStatus.IN_PROGRESS
        active_question = ActiveQuestionContext(
            question_code=next_action.question_code,
            assistant_message_id=assistant_message.message_id,
        )
        previous_pending = session.state.pending_clarification
        previous_context_ids = (
            previous_pending.context_message_ids
            if previous_pending is not None
            and previous_pending.question_code == next_action.question_code
            else ()
        )
        pending_clarification = PendingClarification(
            question_code=next_action.question_code,
            assistant_message_id=assistant_message.message_id,
            reason=clarification_reason,
            context_message_ids=tuple(
                dict.fromkeys((*previous_context_ids, user_message.message_id))
            ),
        )
    elif next_action.action == NextActionType.COMPLETE:
        questionnaire_status = QuestionnaireStatus.COMPLETED
        active_question = None
        pending_clarification = None
    else:
        questionnaire_status = QuestionnaireStatus.COMPLETED_WITH_SKIPS
        active_question = None
        pending_clarification = None

    new_state = QuestionnaireState(
        state_version=session.state.state_version + 1,
        questionnaire_code=session.state.questionnaire_code,
        questionnaire_version=session.state.questionnaire_version,
        status=questionnaire_status,
        questions=tuple(states_by_code[item.question_code] for item in session.state.questions),
        pending_clarification=pending_clarification,
    )
    new_state.validate_against(session.questionnaire)
    return TurnProjection(
        state=new_state,
        active_question=active_question,
        dialog_history=(
            *session.dialog_history,
            user_message,
            assistant_message,
        ),
    )


def _apply_saved_answer(
    previous: QuestionState,
    change: SaveAnswerChange,
    *,
    answer_correction: AnswerCorrectionAuthorization | None,
    active_question_code: str | None,
    max_selections: int,
) -> QuestionState:
    """
    Сохраняет новый ответ или дополняет авторизованный множественный выбор.
    """
    authorized_addition = (
        change.replace_existing
        and answer_correction is not None
        and answer_correction.question_code == change.question_code
        and answer_correction.mode == AnswerCorrectionMode.ADDITIVE
    )
    active_partial_addition = (
        change.question_code == "E1"
        and previous.status == QuestionStatus.PARTIALLY_ANSWERED
        and active_question_code == "E1"
        and not change.replace_existing
    )
    additive = authorized_addition or active_partial_addition
    if previous.status == QuestionStatus.PARTIALLY_ANSWERED and not additive:
        raise ValueError("частичный ответ E1 требует дополнения без потери прежних вариантов")
    if additive:
        old_answer = previous.answer
        if old_answer is None:
            raise ValueError("ADDITIVE требует сохранённого ответа")
        option_codes = tuple(dict.fromkeys((*old_answer.option_codes, *change.option_codes)))
        source_message_ids = tuple(
            dict.fromkeys((*old_answer.source_message_ids, *change.source_message_ids))
        )
        evidence = "; ".join(
            dict.fromkeys(
                fragment
                for fragment in (old_answer.evidence, change.evidence)
                if fragment
            )
        )
    else:
        option_codes = change.option_codes
        source_message_ids = change.source_message_ids
        evidence = change.evidence

    partial_e1 = (
        change.question_code == "E1"
        and active_question_code != "E1"
        and previous.status != QuestionStatus.ANSWERED
        and len(option_codes) < max_selections
    )
    return QuestionState(
        question_code=change.question_code,
        status=(QuestionStatus.PARTIALLY_ANSWERED if partial_e1 else QuestionStatus.ANSWERED),
        answer=SavedAnswer(
            option_codes=option_codes,
            source_message_ids=source_message_ids,
            evidence=evidence,
        ),
        clarification_attempts=0,
    )


def _confirm_partial_answer(
    previous: QuestionState,
    *,
    active_question_code: str | None,
) -> QuestionState:
    """
    Завершает уже начатый выбор без изменения прежних вариантов и оснований.
    """
    if (
        previous.question_code != "E1"
        or previous.status != QuestionStatus.PARTIALLY_ANSWERED
        or previous.answer is None
        or active_question_code != "E1"
    ):
        raise ValueError("CONFIRM_ANSWER допустим только для активного частичного E1")
    return QuestionState(
        question_code=previous.question_code,
        status=QuestionStatus.ANSWERED,
        answer=previous.answer,
    )


def _apply_revised_multi_choice(
    previous: QuestionState,
    change: ReviseMultiChoiceChange,
    *,
    answer_correction: AnswerCorrectionAuthorization | None,
    active_question_code: str | None,
) -> QuestionState:
    """
    Применяет проверенный итоговый выбор или вновь открывает очищенный E1.
    """
    if (
        change.question_code != "E1"
        or previous.status not in {QuestionStatus.ANSWERED, QuestionStatus.PARTIALLY_ANSWERED}
        or previous.answer is None
        or answer_correction is None
        or answer_correction.question_code != "E1"
        or answer_correction.mode != AnswerCorrectionMode.ADDITIVE
        or active_question_code == "E1"
    ):
        raise ValueError("пересмотр выбора E1 не авторизован")
    if not change.option_codes:
        return QuestionState(question_code="E1", status=QuestionStatus.UNKNOWN)
    return QuestionState(
        question_code="E1",
        status=previous.status,
        answer=SavedAnswer(
            option_codes=change.option_codes,
            source_message_ids=change.source_message_ids,
            evidence=change.evidence,
        ),
        clarification_attempts=0,
    )


def _apply_default_answer(
    previous: QuestionState,
    change: ApplyDefaultAnswerChange,
) -> QuestionState:
    """
    Сохраняет ноль по правилу без ссылки на неразобранные реплики ребёнка.
    """
    return QuestionState(
        question_code=change.question_code,
        status=QuestionStatus.ANSWERED,
        answer=SavedAnswer(
            option_codes=change.option_codes,
            origin=AnswerOrigin.POLICY_DEFAULT,
            default_reason=AnswerDefaultReason.CLARIFICATION_LIMIT_REACHED,
        ),
        clarification_attempts=previous.clarification_attempts,
    )


def _apply_clarification(
    previous: QuestionState,
    change: RequestClarificationChange,
) -> QuestionState:
    """
    Уточняет вопрос, сохраняя уже выбранные варианты частичного E1.
    """
    if previous.status == QuestionStatus.PARTIALLY_ANSWERED:
        return QuestionState(
            question_code=change.question_code,
            status=QuestionStatus.PARTIALLY_ANSWERED,
            answer=previous.answer,
            clarification_attempts=previous.clarification_attempts + 1,
        )
    return QuestionState(
        question_code=change.question_code,
        status=QuestionStatus.NEEDS_CLARIFICATION,
        clarification_attempts=previous.clarification_attempts + 1,
    )


def _apply_skip(
    previous: QuestionState,
    change: SkipQuestionChange,
) -> QuestionState:
    """
    Преобразует SKIP_QUESTION, сохраняя число уже выполненных уточнений.
    """
    return QuestionState(
        question_code=change.question_code,
        status=QuestionStatus.SKIPPED,
        clarification_attempts=previous.clarification_attempts,
        skip_reason=change.reason,
    )

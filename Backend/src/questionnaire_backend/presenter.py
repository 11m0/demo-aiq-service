"""
Преобразует внутренний агрегат в безопасный браузерный снимок.
"""

from questionnaire_backend.models import (
    PublicDialogMessage,
    PublicQuestion,
    PublicQuestionnaireAnswer,
    PublicQuestionnaireIdentity,
    PublicQuestionnaireOption,
    PublicSessionSnapshot,
)
from questionnaire_backend.repository import QuestionnaireSession
from questionnaire_contracts import QuestionnaireStatus


def present_session(session: QuestionnaireSession) -> PublicSessionSnapshot:
    """
    Исключает из ответа полный ML-контракт, доказательства и служебный токен.
    """
    states_by_code = {state.question_code: state for state in session.state.questions}
    public_questions: list[PublicQuestion] = []
    for question in session.questionnaire.questions:
        state = states_by_code[question.question_code]
        answer = (
            PublicQuestionnaireAnswer(
                option_codes=state.answer.option_codes,
                evidence=state.answer.evidence,
                origin=state.answer.origin,
                default_reason=state.answer.default_reason,
            )
            if state.answer is not None
            else None
        )
        public_questions.append(
            PublicQuestion(
                question_code=question.question_code,
                question_text=question.question_text,
                status=state.status,
                options=tuple(
                    PublicQuestionnaireOption(
                        option_code=option.option_code,
                        text=option.text,
                        numeric_value=option.numeric_value,
                    )
                    for option in question.options
                ),
                answer=answer,
                skip_reason=state.skip_reason,
            )
        )

    return PublicSessionSnapshot(
        session_id=session.session_id,
        state_version=session.state.state_version,
        status=session.state.status,
        completed=session.state.status != QuestionnaireStatus.IN_PROGRESS,
        dialog_mode=session.dialog_mode,
        active_question_code=(
            session.active_question.question_code if session.active_question is not None else None
        ),
        companion_question_code=(
            session.active_question.companion_question_code
            if session.active_question is not None
            else None
        ),
        questionnaire=PublicQuestionnaireIdentity(
            questionnaire_code=session.questionnaire.questionnaire_code,
            questionnaire_version=session.questionnaire.questionnaire_version,
        ),
        questions=tuple(public_questions),
        dialog_history=tuple(
            PublicDialogMessage(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in session.dialog_history
        ),
    )

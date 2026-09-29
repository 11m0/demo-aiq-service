"""
Описывает полный вход одного хода диалога от backend к ML-сервису.
"""

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from questionnaire_contracts.contract_types import ContractModel, NonEmptyString
from questionnaire_contracts.questionnaire_definition import QuestionnaireDefinition, QuestionType
from questionnaire_contracts.questionnaire_state import QuestionnaireState, QuestionStatus


class DialogRole(StrEnum):
    """
    Перечисляет допустимых авторов сообщений в истории диалога.
    """

    USER = "USER"
    ASSISTANT = "ASSISTANT"


class RespondentMode(StrEnum):
    """
    Показывает, кто отвечает на вопросы выбранной анкеты.
    """

    CHILD = "CHILD"
    CHILD_WITH_PARENT = "CHILD_WITH_PARENT"
    PARENT = "PARENT"


class TurnAnswerScope(StrEnum):
    """
    Ограничивает вопросы, ответы на которые агент вправе извлечь за один ход.
    """

    ACTIVE_QUESTION = "ACTIVE_QUESTION"
    ALL_UNANSWERED = "ALL_UNANSWERED"


class RespondentContext(ContractModel):
    """
    Передаёт минимальный контекст для выбора формулировок диалога.
    """

    age: int = Field(ge=6, le=17)
    mode: RespondentMode
    locale: NonEmptyString = "ru-RU"


class DialogMessage(ContractModel):
    """
    Описывает одну упорядоченную реплику пользователя или ассистента.
    """

    message_id: NonEmptyString
    role: DialogRole
    content: NonEmptyString
    created_at: AwareDatetime


class ActiveQuestionContext(ContractModel):
    """
    Связывает ожидаемый ответ с конкретной репликой ассистента.
    """

    question_code: NonEmptyString
    assistant_message_id: NonEmptyString
    companion_question_code: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_companion(self) -> Self:
        """
        Проверяет, что дополнительный вопрос отличается от основного.
        """
        if self.companion_question_code == self.question_code:
            raise ValueError("companionQuestionCode должен отличаться от questionCode")
        return self


class AnswerCorrectionMode(StrEnum):
    """
    Различает полную замену ответа и добавление вариантов к нему.
    """

    REPLACE = "REPLACE"
    ADDITIVE = "ADDITIVE"


class AnswerCorrectionAuthorization(ContractModel):
    """
    Разрешает агенту обновить один сохранённый ответ в заданном режиме.
    """

    question_code: NonEmptyString
    mode: AnswerCorrectionMode = AnswerCorrectionMode.REPLACE


class TurnRequest(ContractModel):
    """
    Содержит все данные для stateless-обработки одной новой реплики.
    """

    schema_version: NonEmptyString
    correlation_id: NonEmptyString
    session_id: NonEmptyString
    respondent: RespondentContext
    questionnaire: QuestionnaireDefinition
    state: QuestionnaireState
    answer_scope: TurnAnswerScope = TurnAnswerScope.ACTIVE_QUESTION
    active_question: ActiveQuestionContext | None
    answer_correction: AnswerCorrectionAuthorization | None
    replaceable_question_codes: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    dialog_history: tuple[DialogMessage, ...] = Field(default_factory=tuple)
    current_message: DialogMessage
    requested_at: AwareDatetime

    @model_validator(mode="after")
    def validate_turn_context(self) -> Self:
        """
        Проверяет целостность анкеты, состояния и истории текущего хода.
        """
        if self.current_message.role != DialogRole.USER:
            raise ValueError("currentMessage должен содержать новую реплику пользователя")

        history_by_id = {message.message_id: message for message in self.dialog_history}
        if len(history_by_id) != len(self.dialog_history):
            raise ValueError("история диалога не должна повторять идентификаторы сообщений")
        if self.current_message.message_id in history_by_id:
            raise ValueError("currentMessage не должен повторяться в dialogHistory")
        history_positions = {
            message.message_id: position for position, message in enumerate(self.dialog_history)
        }

        self.state.validate_against(self.questionnaire)
        states_by_code = {item.question_code: item for item in self.state.questions}

        pending = self.state.pending_clarification
        if self.answer_scope == TurnAnswerScope.ALL_UNANSWERED:
            if pending is not None:
                raise ValueError("answerScope=ALL_UNANSWERED недопустим при ожидаемом уточнении")
            if self.active_question is not None:
                raise ValueError("answerScope=ALL_UNANSWERED требует activeQuestion=null")
            if self.answer_correction is not None:
                raise ValueError("answerScope=ALL_UNANSWERED требует answerCorrection=null")
            if self.replaceable_question_codes:
                raise ValueError("answerScope=ALL_UNANSWERED требует пустой replaceableQuestionCodes")

        if pending is not None:
            assistant_message = history_by_id.get(pending.assistant_message_id)
            if assistant_message is None:
                raise ValueError("ожидаемое уточнение должно ссылаться на сообщение из истории")
            if assistant_message.role != DialogRole.ASSISTANT:
                raise ValueError("ожидаемое уточнение должно ссылаться на реплику ассистента")

            assistant_position = history_positions[pending.assistant_message_id]
            context_positions: list[int] = []
            for context_message_id in pending.context_message_ids:
                context_message = history_by_id.get(context_message_id)
                if context_message is None:
                    raise ValueError(
                        "контекст ожидаемого уточнения должен ссылаться на сообщения из истории"
                    )
                if context_message.role != DialogRole.USER:
                    raise ValueError(
                        "контекст ожидаемого уточнения должен ссылаться на реплики пользователя"
                    )
                if history_positions[context_message_id] >= assistant_position:
                    raise ValueError(
                        "сообщения контекста должны предшествовать реплике ассистента с уточнением"
                    )
                context_positions.append(history_positions[context_message_id])
            if context_positions != sorted(context_positions):
                raise ValueError(
                    "сообщения контекста уточнения должны идти в хронологическом порядке"
                )

        active_question = self.active_question
        if active_question is not None:
            active_state = states_by_code.get(active_question.question_code)
            if active_state is None:
                raise ValueError("activeQuestion должна ссылаться на вопрос анкеты")

            active_message = history_by_id.get(active_question.assistant_message_id)
            if active_message is None:
                raise ValueError("activeQuestion должна ссылаться на сообщение из истории")
            if active_message.role != DialogRole.ASSISTANT:
                raise ValueError("activeQuestion должна ссылаться на реплику ассистента")
            if self.dialog_history[-1].message_id != active_question.assistant_message_id:
                raise ValueError(
                    "activeQuestion должна ссылаться на последнюю реплику dialogHistory"
                )

            allowed_active_statuses = {
                QuestionStatus.UNKNOWN,
                QuestionStatus.NEEDS_CLARIFICATION,
                QuestionStatus.PARTIALLY_ANSWERED,
            }
            if active_state.status not in allowed_active_statuses:
                raise ValueError("activeQuestion допустима только для незаполненного вопроса")

            companion_code = active_question.companion_question_code
            if companion_code is not None:
                companion_state = states_by_code.get(companion_code)
                if companion_state is None:
                    raise ValueError("companionQuestionCode должен ссылаться на вопрос анкеты")
                if active_state.status != QuestionStatus.UNKNOWN or companion_state.status != QuestionStatus.UNKNOWN:
                    raise ValueError("составной вопрос допустим только для UNKNOWN-вопросов")
                if pending is not None:
                    raise ValueError("составной вопрос недопустим при ожидаемом уточнении")
                composite = next(
                    (
                        item for item in self.questionnaire.composite_questions
                        if item.question_codes == (active_question.question_code, companion_code)
                    ),
                    None,
                )
                if composite is None:
                    raise ValueError("пара activeQuestion не задана в compositeQuestions")
                if active_message.content != composite.question_text:
                    raise ValueError("активная реплика должна совпадать с каноническим составным вопросом")

        if pending is not None:
            if active_question is None:
                raise ValueError("ожидаемое уточнение требует activeQuestion")
            if (
                active_question.question_code != pending.question_code
                or active_question.assistant_message_id != pending.assistant_message_id
            ):
                raise ValueError("activeQuestion должна совпадать с ожидаемым уточнением")

        replaceable_codes = self.replaceable_question_codes
        if len(replaceable_codes) != len(set(replaceable_codes)):
            raise ValueError("replaceableQuestionCodes не должна повторять коды вопросов")
        if replaceable_codes and active_question is None:
            raise ValueError("replaceableQuestionCodes требует activeQuestion")
        questions_by_code = {
            question.question_code: question for question in self.questionnaire.questions
        }
        replaceable_types = {
            QuestionType.SCALE_0_3,
            QuestionType.PAIR_CHOICE,
            QuestionType.SINGLE_CHOICE,
        }
        for question_code in replaceable_codes:
            question = questions_by_code.get(question_code)
            question_state = states_by_code.get(question_code)
            if question is None or question_state is None:
                raise ValueError("replaceableQuestionCodes должна ссылаться на вопрос анкеты")
            if active_question is not None and question_code == active_question.question_code:
                raise ValueError("replaceableQuestionCodes не должна включать activeQuestion")
            if question_state.status != QuestionStatus.ANSWERED:
                raise ValueError("replaceableQuestionCodes допустима только для ANSWERED-вопросов")
            if question.question_type not in replaceable_types:
                raise ValueError("replaceableQuestionCodes допускает только вопросы с одиночным выбором")

        answer_correction = self.answer_correction
        if answer_correction is not None:
            correction_state = states_by_code.get(answer_correction.question_code)
            if correction_state is None:
                raise ValueError("answerCorrection должна ссылаться на вопрос анкеты")
            if correction_state.status not in {
                QuestionStatus.ANSWERED,
                QuestionStatus.PARTIALLY_ANSWERED,
            }:
                raise ValueError("answerCorrection допустима только для сохранённого ответа")
            if (
                correction_state.status == QuestionStatus.PARTIALLY_ANSWERED
                and answer_correction.mode != AnswerCorrectionMode.ADDITIVE
            ):
                raise ValueError("частичный ответ разрешает только ADDITIVE-дополнение")
            if answer_correction.mode == AnswerCorrectionMode.ADDITIVE:
                correction_question = next(
                    question
                    for question in self.questionnaire.questions
                    if question.question_code == answer_correction.question_code
                )
                if correction_question.question_type != QuestionType.MULTI_CHOICE:
                    raise ValueError("ADDITIVE допустим только для MULTI_CHOICE-вопроса")
                if (
                    active_question is None
                    or active_question.question_code == answer_correction.question_code
                ):
                    raise ValueError("ADDITIVE требует другого активного вопроса")

        return self

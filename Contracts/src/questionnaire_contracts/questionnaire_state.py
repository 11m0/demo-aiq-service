"""
Описывает актуальный снимок состояния анкеты, получаемый от backend.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from questionnaire_contracts.contract_types import ContractModel, NonEmptyString
from questionnaire_contracts.questionnaire_definition import (
    QuestionnaireDefinition,
    QuestionType,
)


class QuestionStatus(StrEnum):
    """
    Перечисляет возможные состояния отдельного вопроса.
    """

    UNKNOWN = "UNKNOWN"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    PARTIALLY_ANSWERED = "PARTIALLY_ANSWERED"
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class QuestionnaireStatus(StrEnum):
    """
    Перечисляет состояния прохождения анкеты целиком.
    """

    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_SKIPS = "COMPLETED_WITH_SKIPS"


class AnswerOrigin(StrEnum):
    """
    Показывает, получен ли ответ из реплики пользователя или из бизнес-правила.
    """

    USER_EVIDENCE = "USER_EVIDENCE"
    POLICY_DEFAULT = "POLICY_DEFAULT"


class AnswerDefaultReason(StrEnum):
    """
    Перечисляет допустимые причины автоматического ответа по правилу сервиса.
    """

    CLARIFICATION_LIMIT_REACHED = "CLARIFICATION_LIMIT_REACHED"


class PolicySkipReason(StrEnum):
    """
    Перечисляет причины программного завершения вопроса без ответа.
    """
    NO_ANSWER_AFTER_RETRY = "NO_ANSWER_AFTER_RETRY"
    NO_MATCHING_OPTIONS = "NO_MATCHING_OPTIONS"


class SavedAnswer(ContractModel):
    """
    Хранит подтверждённый backend ответ и сведения о его происхождении.
    """

    option_codes: tuple[NonEmptyString, ...] = Field(min_length=1)
    source_message_ids: tuple[NonEmptyString, ...] = Field(default_factory=tuple)
    evidence: NonEmptyString | None = None
    origin: AnswerOrigin = AnswerOrigin.USER_EVIDENCE
    default_reason: AnswerDefaultReason | None = None

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        """
        Проверяет уникальность и согласованность происхождения ответа.
        """
        if len(self.option_codes) != len(set(self.option_codes)):
            raise ValueError("сохранённый ответ не должен повторять коды вариантов")
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("сохранённый ответ не должен повторять идентификаторы сообщений")

        if self.origin == AnswerOrigin.USER_EVIDENCE:
            if not self.source_message_ids:
                raise ValueError("пользовательский ответ требует хотя бы один источник")
            if self.default_reason is not None:
                raise ValueError(
                    "пользовательский ответ не должен содержать причину автозаполнения"
                )
        else:
            if self.source_message_ids:
                raise ValueError("автоматический ответ не должен ссылаться на реплики пользователя")
            if self.evidence is not None:
                raise ValueError(
                    "автоматический ответ не должен содержать пользовательское основание"
                )
            if self.default_reason is None:
                raise ValueError("автоматический ответ требует причину автозаполнения")
        return self


class QuestionState(ContractModel):
    """
    Описывает состояние одного вопроса в текущей сессии.
    """

    question_code: NonEmptyString
    status: QuestionStatus = QuestionStatus.UNKNOWN
    answer: SavedAnswer | None = None
    clarification_attempts: int = Field(default=0, ge=0)
    skip_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_details(self) -> Self:
        """
        Проверяет соответствие ответа и служебных полей статусу вопроса.
        """
        answered_statuses = {QuestionStatus.ANSWERED, QuestionStatus.PARTIALLY_ANSWERED}
        if self.status == QuestionStatus.ANSWERED and self.answer is None:
            raise ValueError("состояние ANSWERED требует сохранённого ответа")
        if self.status == QuestionStatus.PARTIALLY_ANSWERED and self.answer is None:
            raise ValueError("состояние PARTIALLY_ANSWERED требует сохранённого ответа")
        if self.status not in answered_statuses and self.answer is not None:
            raise ValueError("сохранённый ответ допустим только в заполненном состоянии")
        if (
            self.status == QuestionStatus.PARTIALLY_ANSWERED
            and self.answer is not None
            and self.answer.origin != AnswerOrigin.USER_EVIDENCE
        ):
            raise ValueError("частичный ответ требует пользовательского основания")

        if self.status == QuestionStatus.SKIPPED and self.skip_reason is None:
            raise ValueError("состояние SKIPPED требует причины пропуска")
        if self.status != QuestionStatus.SKIPPED and self.skip_reason is not None:
            raise ValueError("причина пропуска допустима только в состоянии SKIPPED")

        if self.status == QuestionStatus.NEEDS_CLARIFICATION:
            if self.clarification_attempts < 1:
                raise ValueError("ожидаемое уточнение требует хотя бы одной попытки")
        elif self.status == QuestionStatus.PARTIALLY_ANSWERED:
            if self.clarification_attempts > 2:
                raise ValueError("частичный ответ допускает не более двух уточнений")
        elif self.status == QuestionStatus.SKIPPED:
            pass
        elif (
            self.status == QuestionStatus.ANSWERED
            and self.answer is not None
            and self.answer.origin == AnswerOrigin.POLICY_DEFAULT
        ):
            if self.clarification_attempts != 2:
                raise ValueError("автоматический ответ требует ровно двух уточнений")
        elif self.clarification_attempts != 0:
            raise ValueError("счётчик уточнений допустим только для уточнения или пропуска")

        return self


class PendingClarification(ContractModel):
    """
    Связывает ожидаемое уточнение с вопросом и репликой ассистента.
    """

    question_code: NonEmptyString
    assistant_message_id: NonEmptyString
    context_message_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_unique_context_messages(self) -> Self:
        """
        Проверяет отсутствие повторов среди сообщений контекста уточнения.
        """
        if len(self.context_message_ids) != len(set(self.context_message_ids)):
            raise ValueError("контекст уточнения не должен повторять сообщения")
        return self


class QuestionnaireState(ContractModel):
    """
    Хранит полный актуальный снимок прохождения выбранной анкеты.
    """

    state_version: int = Field(ge=0)
    questionnaire_code: NonEmptyString
    questionnaire_version: NonEmptyString
    status: QuestionnaireStatus = QuestionnaireStatus.IN_PROGRESS
    questions: tuple[QuestionState, ...] = Field(min_length=1)
    pending_clarification: PendingClarification | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        """
        Проверяет уникальность вопросов, уточнение и итоговый статус анкеты.
        """
        states_by_code = {item.question_code: item for item in self.questions}
        if len(states_by_code) != len(self.questions):
            raise ValueError("снимок состояния не должен повторять коды вопросов")

        clarification_codes = {
            item.question_code
            for item in self.questions
            if item.status == QuestionStatus.NEEDS_CLARIFICATION
            or (
                item.status == QuestionStatus.PARTIALLY_ANSWERED
                and item.clarification_attempts > 0
            )
        }
        if self.pending_clarification is None and clarification_codes:
            raise ValueError("состояние NEEDS_CLARIFICATION требует ожидаемого уточнения")
        if self.pending_clarification is not None:
            pending_code = self.pending_clarification.question_code
            if clarification_codes != {pending_code}:
                raise ValueError("ожидаемое уточнение должно указывать единственный такой вопрос")

        terminal_statuses = {
            QuestionStatus.ANSWERED,
            QuestionStatus.SKIPPED,
            QuestionStatus.NOT_APPLICABLE,
        }
        all_terminal = all(item.status in terminal_statuses for item in self.questions)
        skipped_count = sum(item.status == QuestionStatus.SKIPPED for item in self.questions)

        if self.status == QuestionnaireStatus.IN_PROGRESS and all_terminal:
            raise ValueError("завершённый набор вопросов не может иметь статус IN_PROGRESS")
        if self.status == QuestionnaireStatus.COMPLETED:
            if not all_terminal or skipped_count:
                raise ValueError("COMPLETED требует завершения всех вопросов без пропусков")
        if self.status == QuestionnaireStatus.COMPLETED_WITH_SKIPS:
            if not all_terminal or not skipped_count:
                raise ValueError(
                    "COMPLETED_WITH_SKIPS требует завершения и хотя бы одного пропуска"
                )

        return self

    def validate_against(self, questionnaire: QuestionnaireDefinition) -> None:
        """
        Проверяет соответствие снимка конкретной версии полной анкеты.
        """
        if self.questionnaire_code != questionnaire.questionnaire_code:
            raise ValueError("код анкеты в состоянии не совпадает с переданной анкетой")
        if self.questionnaire_version != questionnaire.questionnaire_version:
            raise ValueError("версия анкеты в состоянии не совпадает с переданной анкетой")

        questions_by_code = {
            question.question_code: question for question in questionnaire.questions
        }
        states_by_code = {item.question_code: item for item in self.questions}
        if set(states_by_code) != set(questions_by_code):
            raise ValueError(
                "состояние должно содержать каждый вопрос переданной анкеты ровно один раз"
            )

        for question_code, item in states_by_code.items():
            if item.answer is None:
                continue
            question = questions_by_code[question_code]
            if (
                item.status == QuestionStatus.PARTIALLY_ANSWERED
                and question.question_type != QuestionType.MULTI_CHOICE
            ):
                raise ValueError("частичный ответ допустим только для MULTI_CHOICE-вопроса")
            selected_codes = item.answer.option_codes
            if (
                item.status == QuestionStatus.PARTIALLY_ANSWERED
                and len(selected_codes) >= question.max_selections
            ):
                raise ValueError("полный набор вариантов не может оставаться частичным")
            if not set(selected_codes).issubset(question.allowed_option_codes):
                raise ValueError(f"ответ на {question_code} содержит неизвестный вариант")
            if not question.min_selections <= len(selected_codes) <= question.max_selections:
                raise ValueError(f"ответ на {question_code} нарушает ограничение числа вариантов")
            if item.answer.origin == AnswerOrigin.POLICY_DEFAULT:
                zero_option_codes = tuple(
                    option.option_code for option in question.options if option.numeric_value == 0
                )
                if (
                    question.question_type != QuestionType.SCALE_0_3
                    or selected_codes != zero_option_codes
                ):
                    raise ValueError(
                        f"автоматический ответ на {question_code} должен выбирать значение 0"
                    )

"""
Описывает успешный результат или контролируемую ошибку одного хода.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from questionnaire_contracts.contract_types import (
    CanonicalPromptText,
    ContractModel,
    NonEmptyString,
    PreservedNonEmptyString,
)
from questionnaire_contracts.questionnaire_state import AnswerDefaultReason


class SaveAnswerChange(ContractModel):
    """
    Предлагает сохранить новый ответ или явно заменить существующий.
    """

    operation: Literal["SAVE_ANSWER"] = "SAVE_ANSWER"
    question_code: NonEmptyString
    option_codes: tuple[NonEmptyString, ...] = Field(min_length=1)
    source_message_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    evidence: NonEmptyString
    replace_existing: bool = False

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        """
        Проверяет отсутствие повторов в вариантах и источниках ответа.
        """
        if len(self.option_codes) != len(set(self.option_codes)):
            raise ValueError("изменение ответа не должно повторять коды вариантов")
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("изменение ответа не должно повторять источники")
        return self


class ReviseMultiChoiceChange(ContractModel):
    """
    Заменяет сохранённый набор вариантов, в том числе очищает его целиком.
    """

    operation: Literal["REVISE_MULTI_CHOICE"] = "REVISE_MULTI_CHOICE"
    question_code: NonEmptyString
    option_codes: tuple[NonEmptyString, ...]
    source_message_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    evidence: NonEmptyString

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        """
        Запрещает повторённые варианты и ссылки на реплики.
        """
        if len(self.option_codes) != len(set(self.option_codes)):
            raise ValueError("исправленный выбор не должен повторять варианты")
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("исправленный выбор не должен повторять источники")
        return self


class ConfirmAnswerChange(ContractModel):
    """
    Подтверждает завершение ранее начатого множественного ответа.
    """

    operation: Literal["CONFIRM_ANSWER"] = "CONFIRM_ANSWER"
    question_code: NonEmptyString
    source_message_id: NonEmptyString
    evidence: NonEmptyString


class ApplyDefaultAnswerChange(ContractModel):
    """
    Назначает служебный нулевой ответ без пользовательского основания.
    """
    operation: Literal["APPLY_DEFAULT_ANSWER"] = "APPLY_DEFAULT_ANSWER"
    question_code: NonEmptyString
    option_codes: tuple[NonEmptyString, ...] = Field(min_length=1, max_length=1)
    reason: Literal[AnswerDefaultReason.CLARIFICATION_LIMIT_REACHED] = (
        AnswerDefaultReason.CLARIFICATION_LIMIT_REACHED
    )


class RequestClarificationChange(ContractModel):
    """
    Предлагает перевести один вопрос в ожидание уточнения.
    """

    operation: Literal["REQUEST_CLARIFICATION"] = "REQUEST_CLARIFICATION"
    question_code: NonEmptyString
    reason: NonEmptyString
    clarification_text: Annotated[CanonicalPromptText, Field(max_length=240)]


class SkipQuestionChange(ContractModel):
    """
    Предлагает пропустить вопрос после подтверждённого неответа пользователя.
    """

    operation: Literal["SKIP_QUESTION"] = "SKIP_QUESTION"
    question_code: NonEmptyString
    reason: NonEmptyString
    source_message_id: NonEmptyString
    evidence: NonEmptyString


StateChange = Annotated[
    (
        SaveAnswerChange
        | ReviseMultiChoiceChange
        | ConfirmAnswerChange
        | ApplyDefaultAnswerChange
        | RequestClarificationChange
        | SkipQuestionChange
    ),
    Field(discriminator="operation"),
]


class NextActionType(StrEnum):
    """
    Содержит строковые константы следующих действий диалога.
    """

    ASK_QUESTION = "ASK_QUESTION"
    REASK_QUESTION = "REASK_QUESTION"
    CLARIFY = "CLARIFY"
    COMPLETE = "COMPLETE"
    COMPLETE_WITH_SKIPS = "COMPLETE_WITH_SKIPS"


class NextAction(ContractModel):
    """
    Описывает следующее пользовательское действие после применения патча.
    """

    action: NextActionType
    question_code: NonEmptyString | None = None
    companion_question_code: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_question_reference(self) -> Self:
        """
        Проверяет наличие кода вопроса только у действий, которым он нужен.
        """
        actions_with_question = {
            NextActionType.ASK_QUESTION,
            NextActionType.REASK_QUESTION,
            NextActionType.CLARIFY,
        }
        if self.action in actions_with_question and self.question_code is None:
            raise ValueError("действие с вопросом требует questionCode")
        if self.action not in actions_with_question and self.question_code is not None:
            raise ValueError("завершение хода не должно содержать questionCode")
        if self.companion_question_code is not None:
            if self.action != NextActionType.ASK_QUESTION:
                raise ValueError("companionQuestionCode допустим только для ASK_QUESTION")
            if self.companion_question_code == self.question_code:
                raise ValueError("companionQuestionCode должен отличаться от questionCode")
        return self


class TurnResponseBase(ContractModel):
    """
    Хранит идентификаторы исходного запроса и версии его состояния.
    """

    schema_version: NonEmptyString
    correlation_id: NonEmptyString
    session_id: NonEmptyString
    source_message_id: NonEmptyString
    base_state_version: int = Field(ge=0)
    questionnaire_code: NonEmptyString
    questionnaire_version: NonEmptyString
    processed_at: AwareDatetime


class TurnSuccessResponse(TurnResponseBase):
    """
    Возвращает проверяемый патч и следующую реплику ассистента.
    """

    status: Literal["SUCCESS"] = "SUCCESS"
    changes: tuple[StateChange, ...] = Field(default_factory=tuple)
    next_action: NextAction
    assistant_reply: PreservedNonEmptyString

    @model_validator(mode="after")
    def validate_changes_and_action(self) -> Self:
        """
        Проверяет уникальность изменений и связь уточнения со следующим действием.
        """
        changed_codes = tuple(change.question_code for change in self.changes)
        if len(changed_codes) != len(set(changed_codes)):
            raise ValueError("один ход не должен менять один вопрос несколько раз")
        if self.next_action.action == NextActionType.REASK_QUESTION and self.changes:
            if not all(
                isinstance(change, (SaveAnswerChange, ReviseMultiChoiceChange))
                for change in self.changes
            ):
                raise ValueError(
                    "повтор вопроса допускает только сохранение попутных ответов "
                    "и пересмотр выбора"
                )

        clarification_changes = tuple(
            change for change in self.changes if isinstance(change, RequestClarificationChange)
        )
        if self.next_action.action == NextActionType.CLARIFY:
            if len(clarification_changes) != 1:
                raise ValueError("действие CLARIFY требует одного изменения уточнения")
            if clarification_changes[0].question_code != self.next_action.question_code:
                raise ValueError(
                    "изменение уточнения и nextAction должны относиться к одному вопросу"
                )
        elif clarification_changes:
            raise ValueError("REQUEST_CLARIFICATION допустим только вместе с действием CLARIFY")

        return self


class TurnError(ContractModel):
    """
    Описывает безопасную контролируемую ошибку обработки хода.
    """

    code: NonEmptyString
    message: NonEmptyString
    retryable: bool


class TurnFailureResponse(TurnResponseBase):
    """
    Возвращает ошибку без предлагаемых изменений состояния.
    """

    status: Literal["FAILED"] = "FAILED"
    error: TurnError


TurnResponse = Annotated[
    TurnSuccessResponse | TurnFailureResponse,
    Field(discriminator="status"),
]

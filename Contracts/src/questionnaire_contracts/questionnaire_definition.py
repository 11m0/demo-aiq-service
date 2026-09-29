"""
Описывает передаваемую backend анкету и проверяет её структуру.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, FiniteFloat, model_validator

from questionnaire_contracts.contract_types import (
    CanonicalPromptText,
    ContractModel,
    NonEmptyString,
)


class QuestionType(StrEnum):
    """
    Перечисляет поддерживаемые типы вопросов с выбором ответа.
    """

    SCALE_0_3 = "SCALE_0_3"
    SINGLE_CHOICE = "SINGLE_CHOICE"
    MULTI_CHOICE = "MULTI_CHOICE"
    PAIR_CHOICE = "PAIR_CHOICE"


class QuestionOption(ContractModel):
    """
    Описывает один допустимый вариант ответа на вопрос.
    """

    option_code: NonEmptyString
    text: NonEmptyString
    numeric_value: FiniteFloat | None = None


class Question(ContractModel):
    """
    Описывает вопрос и детерминированные ограничения его заполнения.
    """

    question_code: NonEmptyString
    question_type: QuestionType
    question_text: CanonicalPromptText
    answer_meaning: NonEmptyString | None = None
    required: bool = True
    min_selections: int = Field(default=1, ge=0)
    max_selections: int = Field(default=1, ge=1)
    options: tuple[QuestionOption, ...] = Field(min_length=1)
    prompt_hint: NonEmptyString | None = None
    clarification_text: CanonicalPromptText | None = None
    followup_clarification_text: CanonicalPromptText | None = None

    @property
    def allowed_option_codes(self) -> tuple[str, ...]:
        """
        Возвращает коды вариантов в порядке, заданном анкетой.
        """
        return tuple(option.option_code for option in self.options)

    @model_validator(mode="after")
    def validate_selection_rules(self) -> Self:
        """
        Проверяет ограничения выбора и уникальность вариантов ответа.
        """
        if self.min_selections > self.max_selections:
            raise ValueError("minSelections не может превышать maxSelections")
        if self.max_selections > len(self.options):
            raise ValueError("maxSelections не может превышать число вариантов ответа")
        if self.required and self.min_selections == 0:
            raise ValueError("обязательный вопрос должен требовать хотя бы один вариант")

        option_codes = self.allowed_option_codes
        if len(option_codes) != len(set(option_codes)):
            raise ValueError("коды вариантов ответа должны быть уникальными внутри вопроса")

        single_choice_types = {
            QuestionType.SCALE_0_3,
            QuestionType.SINGLE_CHOICE,
            QuestionType.PAIR_CHOICE,
        }
        if self.question_type in single_choice_types and self.max_selections != 1:
            raise ValueError("вопрос с одиночным выбором должен иметь maxSelections=1")

        if self.question_type == QuestionType.SCALE_0_3:
            numeric_values = {option.numeric_value for option in self.options}
            if len(self.options) != 4 or numeric_values != {0.0, 1.0, 2.0, 3.0}:
                raise ValueError("шкала SCALE_0_3 должна содержать значения 0, 1, 2 и 3")

        return self


class CompositeQuestion(ContractModel):
    """
    Связывает два самостоятельных вопроса одной канонической репликой.
    """

    question_codes: tuple[NonEmptyString, NonEmptyString]
    question_text: CanonicalPromptText

    @model_validator(mode="after")
    def validate_distinct_codes(self) -> Self:
        """
        Запрещает дважды спрашивать один и тот же вопрос в общей реплике.
        """
        if self.question_codes[0] == self.question_codes[1]:
            raise ValueError("составной вопрос требует два разных кода")
        return self


class QuestionnaireDefinition(ContractModel):
    """
    Описывает полную выбранную анкету одной версии.
    """

    questionnaire_code: NonEmptyString
    questionnaire_version: NonEmptyString
    questions: tuple[Question, ...] = Field(min_length=1)
    composite_questions: tuple[CompositeQuestion, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_question_codes(self) -> Self:
        """
        Проверяет уникальность кодов вопросов во всей анкете.
        """
        question_codes = tuple(question.question_code for question in self.questions)
        if len(question_codes) != len(set(question_codes)):
            raise ValueError("коды вопросов должны быть уникальными внутри анкеты")
        known_codes = set(question_codes)
        used_composite_codes: set[str] = set()
        for composite in self.composite_questions:
            if not set(composite.question_codes) <= known_codes:
                raise ValueError("составной вопрос должен ссылаться на вопросы анкеты")
            if used_composite_codes.intersection(composite.question_codes):
                raise ValueError("код вопроса не должен повторяться в составных вопросах")
            used_composite_codes.update(composite.question_codes)
        return self

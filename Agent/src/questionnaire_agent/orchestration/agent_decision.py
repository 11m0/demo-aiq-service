"""
Описывает предметное решение модели до добавления служебного envelope.
"""
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

DecisionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]
ClarificationText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]
ExplanationText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=400),
]
TransitionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=120,
        pattern=r"^[^\r\n?？⁇⁈⁉]+$",
    ),
]
OptionReference = Annotated[
    str,
    StringConstraints(pattern=r"^V[1-9][0-9]*$", max_length=12),
]


class AgentIndirectSignal(StrEnum):
    """
    Перечисляет безопасные косвенные признаки ответа на шкалу.
    """

    NEGATIVE_ACTIVITY_ATTITUDE = "NEGATIVE_ACTIVITY_ATTITUDE"
    UNQUALIFIED_ACTIVITY_DISLIKE = "UNQUALIFIED_ACTIVITY_DISLIKE"
    EXPLICIT_ACTIVITY_AVOIDANCE = "EXPLICIT_ACTIVITY_AVOIDANCE"
    POSITIVE_ACTIVITY_ATTITUDE = "POSITIVE_ACTIVITY_ATTITUDE"
    POSITIVE_VIGOROUS_ACTIVITY_ATTITUDE = "POSITIVE_VIGOROUS_ACTIVITY_ATTITUDE"
    NEGATIVE_SCHOOL_PE_ATTITUDE = "NEGATIVE_SCHOOL_PE_ATTITUDE"
    POSITIVE_SUBTYPE_INTEREST = "POSITIVE_SUBTYPE_INTEREST"
    SPONTANEOUS_MUSIC_RECALL = "SPONTANEOUS_MUSIC_RECALL"
    SUSTAINED_CALM_FOCUS = "SUSTAINED_CALM_FOCUS"
    SUSTAINED_HANDCRAFT_ENGAGEMENT = "SUSTAINED_HANDCRAFT_ENGAGEMENT"
    PERSONAL_HANDCRAFT_ACHIEVEMENT = "PERSONAL_HANDCRAFT_ACHIEVEMENT"


class AgentPairSide(StrEnum):
    """
    Указывает предпочитаемую сторону парного вопроса.
    """

    FIRST = "FIRST"
    SECOND = "SECOND"


class AgentPairStrength(StrEnum):
    """
    Указывает обычное или исключительное предпочтение стороны.
    """

    PREFERRED = "PREFERRED"
    EXCLUSIVE = "EXCLUSIVE"


class _StrictAgentModel(BaseModel):
    """
    Задаёт строгие правила внутренних структур, генерируемых моделью.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class AgentSaveAnswerChange(_StrictAgentModel):
    """
    Предлагает сохранить ответ или заменить один разрешённый ответ.
    """

    operation: Literal["SAVE_ANSWER"]
    question_code: DecisionText
    option_codes: tuple[DecisionText, ...] = Field(min_length=1)
    evidence_candidate_id: DecisionText
    replace_existing: bool

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        """
        Проверяет отсутствие повторов вариантов ответа.
        """
        if len(self.option_codes) != len(set(self.option_codes)):
            raise ValueError("изменение ответа не должно повторять коды вариантов")
        return self


class AgentScaleInterpretation(_StrictAgentModel):
    """
    Передаёт прямой уровень шкалы и смысловые косвенные признаки.
    """

    kind: Literal["SCALE"]
    direct_level: Annotated[int, Field(strict=True, ge=0, le=3)] | None
    indirect_signals: tuple[AgentIndirectSignal, ...]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """
        Требует хотя бы один способ интерпретации и запрещает повторы.
        """
        if self.direct_level is None and not self.indirect_signals:
            raise ValueError("интерпретация шкалы должна содержать уровень или косвенный признак")
        if len(self.indirect_signals) != len(set(self.indirect_signals)):
            raise ValueError("интерпретация шкалы не должна повторять косвенные признаки")
        return self


class AgentPairPreferenceInterpretation(_StrictAgentModel):
    """
    Передаёт сторону и силу предпочтения в парном вопросе.
    """

    kind: Literal["PAIR_PREFERENCE"]
    side: AgentPairSide
    strength: AgentPairStrength


class AgentOptionSelectionInterpretation(_StrictAgentModel):
    """
    Передаёт позиции выбранных вариантов без бизнес-кодов анкеты.
    """

    kind: Literal["OPTION_SELECTION"]
    option_refs: tuple[OptionReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_references(self) -> Self:
        """
        Запрещает повторять одну позиционную ссылку.
        """
        if len(self.option_refs) != len(set(self.option_refs)):
            raise ValueError("интерпретация выбора не должна повторять ссылки на варианты")
        return self


AgentAnswerInterpretation = Annotated[
    AgentScaleInterpretation
    | AgentPairPreferenceInterpretation
    | AgentOptionSelectionInterpretation,
    Field(discriminator="kind"),
]


class AgentInterpretAnswerChange(_StrictAgentModel):
    """
    Предлагает смысл ответа без прямого доступа к бизнес-кодам вариантов.
    """

    operation: Literal["INTERPRET_ANSWER"]
    question_code: DecisionText
    interpretation: AgentAnswerInterpretation
    evidence_candidate_id: DecisionText
    replace_existing: bool


class AgentClarification(_StrictAgentModel):
    """
    Предлагает контекстное уточнение одного связанного с репликой вопроса.
    """

    question_code: DecisionText
    reason: DecisionText
    clarification_text: ClarificationText


class AgentExplanation(_StrictAgentModel):
    """
    Объясняет активный вопрос без расходования попытки уточнения.
    """

    question_code: DecisionText
    explanation_text: ExplanationText


class AgentSkipQuestionChange(_StrictAgentModel):
    """
    Предлагает пропустить активный вопрос по явной реплике пользователя.
    """

    operation: Literal["SKIP_QUESTION"]
    question_code: DecisionText
    reason: DecisionText
    evidence_candidate_id: DecisionText


class AgentConfirmAnswerChange(_StrictAgentModel):
    """
    Завершает ранее выбранные занятия без изменения вариантов ответа.
    """

    operation: Literal["CONFIRM_ANSWER"]
    question_code: DecisionText
    evidence_candidate_id: DecisionText


AgentStateChange = (
    AgentSaveAnswerChange
    | AgentInterpretAnswerChange
    | AgentSkipQuestionChange
    | AgentConfirmAnswerChange
)


class AgentQuestionDraft(_StrictAgentModel):
    """
    Предлагает подводку и следующий вопрос для проверки после применения патча.
    """

    question_code: DecisionText
    bridge_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    ]
    question_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
    ]
    asked_question_codes: tuple[DecisionText, ...] = Field(min_length=1, max_length=2)


class AgentDecision(_StrictAgentModel):
    """
    Содержит изменения, уточнение и предпочтение следующего вопроса.
    """

    changes: tuple[AgentStateChange, ...] = Field(max_length=100)
    clarification: AgentClarification | None
    transition_text: TransitionText | None
    explanation: AgentExplanation | None = None
    preferred_next_question_code: DecisionText | None = None
    question_draft: AgentQuestionDraft | None = None

    @model_validator(mode="after")
    def validate_unique_question_targets(self) -> Self:
        """
        Запрещает одновременно сохранять и уточнять один вопрос.
        """
        changed_codes = tuple(change.question_code for change in self.changes)
        if len(changed_codes) != len(set(changed_codes)):
            raise ValueError("решение не должно менять один вопрос несколько раз")
        if self.clarification is not None and self.clarification.question_code in changed_codes:
            raise ValueError("решение не должно одновременно сохранять и уточнять один вопрос")
        if self.clarification is not None and self.transition_text is not None:
            raise ValueError("уточнение не должно содержать отдельную подводку")
        if self.explanation is not None:
            if self.clarification is not None or self.transition_text is not None:
                raise ValueError("пояснение не совмещается с уточнением или подводкой")
            if self.explanation.question_code in changed_codes:
                raise ValueError("поясняемый вопрос не должен одновременно получать оценку")
        return self

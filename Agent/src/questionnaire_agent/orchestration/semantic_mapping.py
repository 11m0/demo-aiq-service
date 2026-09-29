"""
Преобразует смысловой ответ модели в проверяемое сохранение бизнес-варианта.
"""

from enum import StrEnum

from questionnaire_agent.orchestration.agent_decision import (
    AgentIndirectSignal,
    AgentInterpretAnswerChange,
    AgentOptionSelectionInterpretation,
    AgentPairPreferenceInterpretation,
    AgentPairSide,
    AgentPairStrength,
    AgentSaveAnswerChange,
    AgentScaleInterpretation,
)
from questionnaire_contracts import Question, QuestionType

_INDIRECT_SCALE_LEVELS = {
    ("SELF_RATED_SKILL", AgentIndirectSignal.NEGATIVE_ACTIVITY_ATTITUDE): 1,
    ("SELF_RATED_SKILL", AgentIndirectSignal.UNQUALIFIED_ACTIVITY_DISLIKE): 0,
    ("SELF_RATED_SKILL", AgentIndirectSignal.EXPLICIT_ACTIVITY_AVOIDANCE): 0,
    ("SELF_RATED_SKILL", AgentIndirectSignal.POSITIVE_ACTIVITY_ATTITUDE): 2,
    (
        "ENDURANCE_SELF_RATING",
        AgentIndirectSignal.POSITIVE_VIGOROUS_ACTIVITY_ATTITUDE,
    ): 3,
    (
        "ENDURANCE_SELF_RATING",
        AgentIndirectSignal.NEGATIVE_SCHOOL_PE_ATTITUDE,
    ): 1,
    (
        "VISUAL_ATTENTION_FREQUENCY",
        AgentIndirectSignal.POSITIVE_SUBTYPE_INTEREST,
    ): 2,
    ("FREQUENCY", AgentIndirectSignal.SPONTANEOUS_MUSIC_RECALL): 2,
    ("FOCUS_DURATION", AgentIndirectSignal.SUSTAINED_CALM_FOCUS): 2,
    ("HANDCRAFT_ATTITUDE", AgentIndirectSignal.SUSTAINED_HANDCRAFT_ENGAGEMENT): 3,
    ("HANDCRAFT_ATTITUDE", AgentIndirectSignal.PERSONAL_HANDCRAFT_ACHIEVEMENT): 2,
}

_PAIR_OPTION_INDICES = {
    (AgentPairSide.FIRST, AgentPairStrength.EXCLUSIVE): 0,
    (AgentPairSide.FIRST, AgentPairStrength.PREFERRED): 1,
    (AgentPairSide.SECOND, AgentPairStrength.PREFERRED): 2,
    (AgentPairSide.SECOND, AgentPairStrength.EXCLUSIVE): 3,
}


class SemanticMappingErrorCode(StrEnum):
    """
    Перечисляет причины отказа от проекции смыслового ответа.
    """

    QUESTION_CODE_MISMATCH = "QUESTION_CODE_MISMATCH"
    QUESTION_TYPE_MISMATCH = "QUESTION_TYPE_MISMATCH"
    UNSUPPORTED_INDIRECT_SIGNAL = "UNSUPPORTED_INDIRECT_SIGNAL"
    CONFLICTING_INDIRECT_SIGNALS = "CONFLICTING_INDIRECT_SIGNALS"
    OPTION_VALUE_NOT_FOUND = "OPTION_VALUE_NOT_FOUND"
    INVALID_PAIR_LAYOUT = "INVALID_PAIR_LAYOUT"
    INVALID_OPTION_REFERENCE = "INVALID_OPTION_REFERENCE"
    INVALID_SELECTION_COUNT = "INVALID_SELECTION_COUNT"


class SemanticMappingError(ValueError):
    """
    Сообщает типизированную причину невозможной или небезопасной проекции.
    """

    def __init__(self, code: SemanticMappingErrorCode, message: str) -> None:
        """
        Сохраняет машинный код вместе с безопасным описанием ошибки.
        """
        self.code = code
        super().__init__(message)


def map_semantic_answer(
    question: Question,
    change: AgentInterpretAnswerChange,
) -> AgentSaveAnswerChange:
    """
    Проецирует типизированный смысл ответа на коды вариантов одного вопроса.
    """
    if question.question_code != change.question_code:
        raise SemanticMappingError(
            SemanticMappingErrorCode.QUESTION_CODE_MISMATCH,
            "код интерпретации не совпадает с кодом вопроса",
        )

    interpretation = change.interpretation
    if isinstance(interpretation, AgentScaleInterpretation):
        option_codes = _map_scale_interpretation(question, interpretation)
    elif isinstance(interpretation, AgentPairPreferenceInterpretation):
        option_codes = _map_pair_interpretation(question, interpretation)
    elif isinstance(interpretation, AgentOptionSelectionInterpretation):
        option_codes = _map_option_selection(question, interpretation)
    else:
        raise AssertionError("неподдерживаемый тип смысловой интерпретации")

    return AgentSaveAnswerChange(
        operation="SAVE_ANSWER",
        question_code=change.question_code,
        option_codes=option_codes,
        evidence_candidate_id=change.evidence_candidate_id,
        replace_existing=change.replace_existing,
    )


def _map_scale_interpretation(
    question: Question,
    interpretation: AgentScaleInterpretation,
) -> tuple[str, ...]:
    """
    Преобразует прямой уровень или разрешённый косвенный признак шкалы.
    """
    if question.question_type != QuestionType.SCALE_0_3:
        raise SemanticMappingError(
            SemanticMappingErrorCode.QUESTION_TYPE_MISMATCH,
            "интерпретация SCALE допустима только для вопроса SCALE_0_3",
        )

    if interpretation.direct_level is not None:
        level = interpretation.direct_level
    else:
        levels: set[int] = set()
        for signal in interpretation.indirect_signals:
            mapped_level = _INDIRECT_SCALE_LEVELS.get((question.answer_meaning, signal))
            if mapped_level is None:
                raise SemanticMappingError(
                    SemanticMappingErrorCode.UNSUPPORTED_INDIRECT_SIGNAL,
                    "косвенный признак не разрешён для смысла этого вопроса",
                )
            levels.add(mapped_level)
        if len(levels) != 1:
            raise SemanticMappingError(
                SemanticMappingErrorCode.CONFLICTING_INDIRECT_SIGNALS,
                "косвенные признаки задают несовместимые уровни шкалы",
            )
        level = levels.pop()

    matching_options = tuple(
        option.option_code for option in question.options if option.numeric_value == level
    )
    if len(matching_options) != 1:
        raise SemanticMappingError(
            SemanticMappingErrorCode.OPTION_VALUE_NOT_FOUND,
            "уровню шкалы не соответствует ровно один вариант вопроса",
        )
    return matching_options


def _map_pair_interpretation(
    question: Question,
    interpretation: AgentPairPreferenceInterpretation,
) -> tuple[str, ...]:
    """
    Выбирает одну из четырёх упорядоченных позиций парного вопроса.
    """
    if question.question_type != QuestionType.PAIR_CHOICE:
        raise SemanticMappingError(
            SemanticMappingErrorCode.QUESTION_TYPE_MISMATCH,
            "интерпретация PAIR_PREFERENCE допустима только для PAIR_CHOICE",
        )
    if len(question.options) != 4:
        raise SemanticMappingError(
            SemanticMappingErrorCode.INVALID_PAIR_LAYOUT,
            "парный вопрос должен содержать четыре упорядоченных варианта",
        )

    option_index = _PAIR_OPTION_INDICES[(interpretation.side, interpretation.strength)]
    return (question.options[option_index].option_code,)


def _map_option_selection(
    question: Question,
    interpretation: AgentOptionSelectionInterpretation,
) -> tuple[str, ...]:
    """
    Разрешает ссылки V1 и далее по позициям и проверяет число выборов.
    """
    if question.question_type not in {
        QuestionType.SINGLE_CHOICE,
        QuestionType.MULTI_CHOICE,
    }:
        raise SemanticMappingError(
            SemanticMappingErrorCode.QUESTION_TYPE_MISMATCH,
            "интерпретация OPTION_SELECTION допустима для обычного выбора",
        )

    selection_count = len(interpretation.option_refs)
    if not question.min_selections <= selection_count <= question.max_selections:
        raise SemanticMappingError(
            SemanticMappingErrorCode.INVALID_SELECTION_COUNT,
            "число выбранных ссылок не соответствует ограничениям вопроса",
        )

    option_codes: list[str] = []
    for option_ref in interpretation.option_refs:
        option_index = int(option_ref[1:]) - 1
        if option_index >= len(question.options):
            raise SemanticMappingError(
                SemanticMappingErrorCode.INVALID_OPTION_REFERENCE,
                "ссылка указывает за пределы вариантов вопроса",
            )
        option_codes.append(question.options[option_index].option_code)
    return tuple(option_codes)

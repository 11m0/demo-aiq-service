"""
Считает шкалы анкеты для самостоятельного прохождения ребёнком 6–9 лет.
"""

from questionnaire_contracts import (
    QuestionnaireDefinition,
    QuestionnaireState,
    QuestionStatus,
    QuestionType,
)
from questionnaire_core.scoring.models import (
    QuestionnaireScoringResult,
    ScaleCode,
    ScaleScore,
)

_QUESTIONNAIRE_CODE = "AGE_6_9"

_SCALE_LAYOUT = (
    (ScaleCode.HEARING, "Слух", 4),
    (ScaleCode.VISION, "Зрение", 4),
    (ScaleCode.KINESTHETICS, "Кинестетика", 4),
    (ScaleCode.FINE_MOTOR, "Мелкая моторика", 4),
    (ScaleCode.GROSS_MOTOR, "Крупная моторика", 4),
    (ScaleCode.STAGE_PRESENCE, "Сценичность", 4),
    (ScaleCode.INDIVIDUAL_WORK, "Индивидуальная работа", 6),
    (ScaleCode.ENSEMBLE, "Ансамбль", 4),
    (ScaleCode.DISCIPLINE, "Дисциплина", 4),
    (ScaleCode.TECHNOLOGY, "Технологичность", 4),
    (ScaleCode.HANDCRAFT, "Ручной труд", 4),
    (ScaleCode.IMPROVISATION, "Импровизация", 1),
)

_DIRECT_SCALE_BY_QUESTION = {
    "S1": ScaleCode.HEARING,
    "S4": ScaleCode.VISION,
    "S7": ScaleCode.KINESTHETICS,
    "M1": ScaleCode.FINE_MOTOR,
    "M3": ScaleCode.GROSS_MOTOR,
    "T1": ScaleCode.STAGE_PRESENCE,
    "T3": ScaleCode.INDIVIDUAL_WORK,
    "C1": ScaleCode.DISCIPLINE,
    "X1": ScaleCode.TECHNOLOGY,
    "X4": ScaleCode.HANDCRAFT,
}

_T4_SCORES = {
    "O0": {ScaleCode.ENSEMBLE: 0, ScaleCode.INDIVIDUAL_WORK: 3},
    "O1": {ScaleCode.ENSEMBLE: 1, ScaleCode.INDIVIDUAL_WORK: 2},
    "O2": {ScaleCode.ENSEMBLE: 2, ScaleCode.INDIVIDUAL_WORK: 1},
    "O3": {ScaleCode.ENSEMBLE: 3, ScaleCode.INDIVIDUAL_WORK: 0},
}

_E1_CONTRIBUTIONS = {
    "SING": frozenset({ScaleCode.HEARING, ScaleCode.STAGE_PRESENCE}),
    "DANCE": frozenset(
        {
            ScaleCode.KINESTHETICS,
            ScaleCode.GROSS_MOTOR,
            ScaleCode.STAGE_PRESENCE,
        }
    ),
    "DRAW": frozenset({ScaleCode.VISION, ScaleCode.FINE_MOTOR}),
    "CRAFT": frozenset({ScaleCode.FINE_MOTOR, ScaleCode.HANDCRAFT}),
    "INSTRUMENT": frozenset({ScaleCode.HEARING, ScaleCode.FINE_MOTOR, ScaleCode.DISCIPLINE}),
    "PHOTO_VIDEO": frozenset({ScaleCode.VISION, ScaleCode.TECHNOLOGY}),
    "ROLE": frozenset({ScaleCode.STAGE_PRESENCE, ScaleCode.IMPROVISATION, ScaleCode.ENSEMBLE}),
    "DIGITAL": frozenset({ScaleCode.TECHNOLOGY, ScaleCode.VISION}),
    "CONSTRUCTOR": frozenset({ScaleCode.VISION, ScaleCode.DISCIPLINE, ScaleCode.FINE_MOTOR}),
}

_EXPECTED_QUESTION_CODES = frozenset({*_DIRECT_SCALE_BY_QUESTION, "T4", "E1"})
_EXPECTED_SCALE_OPTIONS = {
    "O0": 0.0,
    "O1": 1.0,
    "O2": 2.0,
    "O3": 3.0,
}


class ScoringError(ValueError):
    """
    Служит базовой ошибкой детерминированного скоринга.
    """


class UnsupportedQuestionnaireError(ScoringError):
    """
    Сообщает, что функция получила анкету другого типа.
    """


class IncompatibleScoringInputError(ScoringError):
    """
    Сообщает о несовместимой схеме анкеты или снимке состояния.
    """


def score_age_6_9(
    questionnaire: QuestionnaireDefinition,
    state: QuestionnaireState,
) -> QuestionnaireScoringResult:
    """
    Рассчитывает все шкалы AGE_6_9 по подтверждённым ответам состояния.
    """
    _validate_questionnaire(questionnaire)
    _validate_state(questionnaire, state)

    questions_by_code = {question.question_code: question for question in questionnaire.questions}
    totals = {scale_code: 0 for scale_code, _, _ in _SCALE_LAYOUT}

    for question_state in state.questions:
        if question_state.status != QuestionStatus.ANSWERED:
            continue
        if question_state.answer is None:
            raise IncompatibleScoringInputError(
                f"ответ на {question_state.question_code} не содержит выбранных вариантов"
            )

        question_code = question_state.question_code
        selected_codes = question_state.answer.option_codes
        if question_code in _DIRECT_SCALE_BY_QUESTION:
            question = questions_by_code[question_code]
            options_by_code = {option.option_code: option for option in question.options}
            numeric_value = options_by_code[selected_codes[0]].numeric_value
            if numeric_value is None:
                raise IncompatibleScoringInputError(
                    f"вариант {selected_codes[0]} вопроса {question_code} не содержит балл"
                )
            totals[_DIRECT_SCALE_BY_QUESTION[question_code]] += int(numeric_value)
        elif question_code == "T4":
            for scale_code, score in _T4_SCORES[selected_codes[0]].items():
                totals[scale_code] += score
        elif question_code == "E1":
            affected_scales = {
                scale_code
                for option_code in selected_codes
                for scale_code in _E1_CONTRIBUTIONS[option_code]
            }
            for scale_code in affected_scales:
                totals[scale_code] += 1

    return QuestionnaireScoringResult(
        questionnaire_code=questionnaire.questionnaire_code,
        questionnaire_version=questionnaire.questionnaire_version,
        state_version=state.state_version,
        scales=tuple(
            ScaleScore(
                scale_code=scale_code,
                name=name,
                score=totals[scale_code],
                max_score=max_score,
            )
            for scale_code, name, max_score in _SCALE_LAYOUT
        ),
    )


def _validate_questionnaire(questionnaire: QuestionnaireDefinition) -> None:
    """
    Проверяет код и структуру вопросов, от которых зависит формула скоринга.
    """
    if questionnaire.questionnaire_code != _QUESTIONNAIRE_CODE:
        raise UnsupportedQuestionnaireError(
            f"скоринг поддерживает только анкету {_QUESTIONNAIRE_CODE}"
        )

    questions_by_code = {question.question_code: question for question in questionnaire.questions}
    if set(questions_by_code) != _EXPECTED_QUESTION_CODES:
        raise IncompatibleScoringInputError(
            "состав вопросов AGE_6_9 несовместим с формулой скоринга"
        )

    for question_code in _DIRECT_SCALE_BY_QUESTION:
        question = questions_by_code[question_code]
        actual_options = {option.option_code: option.numeric_value for option in question.options}
        if (
            question.question_type != QuestionType.SCALE_0_3
            or actual_options != _EXPECTED_SCALE_OPTIONS
        ):
            raise IncompatibleScoringInputError(f"вопрос {question_code} несовместим со шкалой 0–3")

    t4 = questions_by_code["T4"]
    if t4.question_type != QuestionType.PAIR_CHOICE or set(t4.allowed_option_codes) != set(
        _T4_SCORES
    ):
        raise IncompatibleScoringInputError("вопрос T4 несовместим с парным скорингом")

    e1 = questions_by_code["E1"]
    if (
        e1.question_type != QuestionType.MULTI_CHOICE
        or set(e1.allowed_option_codes) != set(_E1_CONTRIBUTIONS)
        or e1.min_selections != 1
        or e1.max_selections != len(_E1_CONTRIBUTIONS)
    ):
        raise IncompatibleScoringInputError("вопрос E1 несовместим со скорингом выбранных занятий")


def _validate_state(
    questionnaire: QuestionnaireDefinition,
    state: QuestionnaireState,
) -> None:
    """
    Проверяет принадлежность и содержимое состояния перед расчётом.
    """
    try:
        state.validate_against(questionnaire)
    except ValueError as error:
        raise IncompatibleScoringInputError(f"состояние несовместимо с анкетой: {error}") from error

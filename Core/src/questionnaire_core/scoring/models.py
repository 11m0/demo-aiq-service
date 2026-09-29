"""
Описывает неизменяемый результат детерминированного скоринга анкеты.
"""

from dataclasses import dataclass
from enum import StrEnum


class ScaleCode(StrEnum):
    """
    Перечисляет стабильные машинные коды шкал анкеты.
    """

    HEARING = "HEARING"
    VISION = "VISION"
    KINESTHETICS = "KINESTHETICS"
    FINE_MOTOR = "FINE_MOTOR"
    GROSS_MOTOR = "GROSS_MOTOR"
    STAGE_PRESENCE = "STAGE_PRESENCE"
    INDIVIDUAL_WORK = "INDIVIDUAL_WORK"
    ENSEMBLE = "ENSEMBLE"
    DISCIPLINE = "DISCIPLINE"
    TECHNOLOGY = "TECHNOLOGY"
    HANDCRAFT = "HANDCRAFT"
    IMPROVISATION = "IMPROVISATION"


@dataclass(frozen=True, slots=True)
class ScaleScore:
    """
    Хранит итог и максимальное значение одной шкалы.
    """

    scale_code: ScaleCode
    name: str
    score: int
    max_score: int


@dataclass(frozen=True, slots=True)
class QuestionnaireScoringResult:
    """
    Хранит воспроизводимый результат скоринга одного снимка анкеты.
    """

    questionnaire_code: str
    questionnaire_version: str
    state_version: int
    scales: tuple[ScaleScore, ...]

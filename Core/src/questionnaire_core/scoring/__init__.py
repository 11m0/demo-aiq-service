"""
Предоставляет детерминированный скоринг поддерживаемых анкет.
"""

from questionnaire_core.scoring.age_6_9 import (
    IncompatibleScoringInputError,
    ScoringError,
    UnsupportedQuestionnaireError,
    score_age_6_9,
)
from questionnaire_core.scoring.models import (
    QuestionnaireScoringResult,
    ScaleCode,
    ScaleScore,
)

__all__ = [
    "IncompatibleScoringInputError",
    "QuestionnaireScoringResult",
    "ScaleCode",
    "ScaleScore",
    "ScoringError",
    "UnsupportedQuestionnaireError",
    "score_age_6_9",
]

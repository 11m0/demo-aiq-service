"""
Объединяет HTTP-маршруты ML-сервиса.
"""

from questionnaire_api.routes.turns import router as turns_router

__all__ = ["turns_router"]

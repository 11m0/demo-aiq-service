"""
Задаёт стабильные коды ошибок программной обработки хода.
"""
from enum import StrEnum


class TurnProcessingErrorCode(StrEnum):
    """
    Перечисляет безопасные причины отклонения контекста или результата модели.
    """
    CONTEXT_TOO_LARGE = "AGENT_CONTEXT_TOO_LARGE"
    OUTPUT_TOO_LARGE = "AGENT_OUTPUT_TOO_LARGE"
    INVALID_AGENT_OUTPUT = "AGENT_OUTPUT_INVALID"
    REJECTED_AGENT_OUTPUT = "AGENT_OUTPUT_REJECTED"

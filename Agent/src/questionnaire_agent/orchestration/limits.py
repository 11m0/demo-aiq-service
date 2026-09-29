"""
Ограничивает объём данных на границе вызова языковой модели.
"""

import json
from dataclasses import dataclass

from questionnaire_agent.llm import LLMCompletionRequest

DEFAULT_MAX_LLM_REQUEST_BYTES = 512 * 1024
DEFAULT_MAX_LLM_OUTPUT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class TurnProcessingLimits:
    """
    Хранит детерминированные пределы полного запроса и сырого ответа LLM.
    """

    max_llm_request_bytes: int = DEFAULT_MAX_LLM_REQUEST_BYTES
    max_llm_output_bytes: int = DEFAULT_MAX_LLM_OUTPUT_BYTES

    def __post_init__(self) -> None:
        """
        Требует положительные целочисленные значения обоих лимитов.
        """
        values = {
            "max_llm_request_bytes": self.max_llm_request_bytes,
            "max_llm_output_bytes": self.max_llm_output_bytes,
        }
        for field_name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} должен быть положительным целым числом")


def measure_llm_request_bytes(request: LLMCompletionRequest) -> int:
    """
    Считает UTF-8 размер канонического представления полного LLM-запроса.
    """
    payload = {
        "messages": [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in request.messages
        ],
        "responseSchema": {
            "name": request.response_schema.name,
            "strict": request.response_schema.strict,
            "schema": request.response_schema.schema,
        },
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(serialized.encode("utf-8"))


def measure_llm_output_bytes(content: str) -> int:
    """
    Считает UTF-8 размер сырого текста до его JSON-разбора.
    """
    return len(content.encode("utf-8"))

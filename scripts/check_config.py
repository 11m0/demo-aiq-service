"""
Проверяет настройки сервисов без вызовов LLM и без вывода секретных значений.
"""

import argparse
import sys

from pydantic import ValidationError

from questionnaire_api.config import Settings
from questionnaire_backend.config import BackendSettings


def main() -> int:
    """
    Проверяет обязательные настройки и согласованность бюджетов ожидания.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--health-only", action="store_true")
    arguments = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        print("Для воспроизводимого стенда требуется Python 3.12.", file=sys.stderr)
        return 1
    try:
        api = Settings()
        backend = BackendSettings()
    except ValidationError as error:
        # Не печатаем str(error): валидаторы могут содержать входные значения.
        fields = sorted({".".join(map(str, item["loc"])) or "settings" for item in error.errors()})
        print("Некорректные настройки: " + ", ".join(fields), file=sys.stderr)
        return 1
    if not api.llm_configured and not arguments.health_only:
        print("Настройте LLM_API_KEY и LLM_MODEL либо укажите --health-only.", file=sys.stderr)
        return 1
    if api.llm_configured and backend.ml_timeout_seconds <= 3 * api.llm_timeout_seconds:
        print("ML_TIMEOUT_SECONDS должен быть больше 3 * LLM_TIMEOUT_SECONDS.", file=sys.stderr)
        return 1
    print("Конфигурация: " + ("LLM настроен (доступность не проверена)." if api.llm_configured
                              else "health-only; обработка ответов недоступна."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

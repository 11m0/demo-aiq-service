"""
Загружает и проверяет серверные настройки backend-приложения.
"""

import os
from functools import lru_cache
from ipaddress import ip_address
from secrets import compare_digest
from string import ascii_letters, digits
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_TOKEN_CHARACTERS = frozenset(ascii_letters + digits + "-._~+/=")


class BackendSettings(BaseSettings):
    """
    Хранит настройки сессий и защищённого соединения с ML-сервисом.
    """

    service_auth_token: SecretStr
    ml_turn_url: str = "http://127.0.0.1:8000/v1/turns/process"
    ml_timeout_seconds: float = Field(default=200.0, gt=0, le=390, allow_inf_nan=False)
    ml_response_max_bytes: int = Field(default=256 * 1024, ge=1024, le=10 * 1024 * 1024)
    backend_max_request_bytes: int = Field(default=16 * 1024, ge=1024, le=1024 * 1024)
    backend_session_ttl_seconds: int = Field(default=3600, ge=1, le=86_400)
    backend_max_sessions: int = Field(default=1000, ge=1, le=100_000)

    model_config = SettingsConfigDict(
        env_file=os.environ.get("QUESTIONNAIRE_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("service_auth_token")
    @classmethod
    def validate_service_auth_token(cls, value: SecretStr) -> SecretStr:
        """
        Проверяет формат общего service-to-service токена без его раскрытия.
        """
        secret = value.get_secret_value()
        if not 32 <= len(secret) <= 512:
            raise ValueError("Сервисный токен должен содержать от 32 до 512 символов")
        if not secret.isascii() or any(
            character not in _SERVICE_TOKEN_CHARACTERS for character in secret
        ):
            raise ValueError("Сервисный токен содержит недопустимые символы")
        return value

    @field_validator("ml_turn_url")
    @classmethod
    def validate_ml_turn_url(cls, value: str) -> str:
        """
        Принимает абсолютный HTTP(S) URL без встроенных учётных данных.
        """
        if value != value.strip() or not value.isprintable() or "\\" in value:
            raise ValueError("Адрес ML-сервиса задан некорректно")
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError:
            raise ValueError("Адрес ML-сервиса задан некорректно") from None
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("Адрес ML-сервиса должен быть абсолютным HTTP(S) URL")
        if "%" in parsed.netloc or parsed.port == 0:
            raise ValueError("Адрес ML-сервиса содержит недопустимый host или port")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Адрес ML-сервиса не должен содержать учётные данные")
        if parsed.query or parsed.fragment:
            raise ValueError("Адрес ML-сервиса не должен содержать query или fragment")
        if parsed.path != "/v1/turns/process":
            raise ValueError("Адрес ML-сервиса должен вести на /v1/turns/process")
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
            raise ValueError("Обычный HTTP для ML-сервиса разрешён только на loopback")
        return value

    @field_validator(
        "ml_response_max_bytes",
        "backend_max_request_bytes",
        "backend_session_ttl_seconds",
        "backend_max_sessions",
        mode="before",
    )
    @classmethod
    def validate_integer_settings(cls, value: object) -> object:
        """
        Запрещает неявное преобразование логических значений в числовые лимиты.
        """
        if isinstance(value, bool):
            raise ValueError("Числовой лимит не может быть логическим значением")
        return value

    def token_matches(self, candidate: str) -> bool:
        """
        Сравнивает токен без ветвления по совпавшим символам.
        """
        return compare_digest(self.service_auth_token.get_secret_value(), candidate)


@lru_cache
def get_backend_settings() -> BackendSettings:
    """
    Возвращает единый проверенный набор настроек backend-процесса.
    """
    return BackendSettings()


def _is_loopback_host(hostname: str | None) -> bool:
    """
    Распознаёт localhost и loopback IP-адрес без DNS-запроса.
    """
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False

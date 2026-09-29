"""
Загружает и проверяет настройки HTTP-приложения ML-сервиса.
"""

import os
from functools import lru_cache
from ipaddress import ip_address
from secrets import compare_digest
from string import ascii_letters, digits
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_TOKEN_CHARACTERS = frozenset(ascii_letters + digits + "-._~+/=")


class Settings(BaseSettings):
    """
    Хранит настройки приложения и подключения к языковой модели.
    """

    app_env: Literal["local", "dev", "test", "prod"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_max_request_bytes: int = Field(default=1024 * 1024, ge=1, le=10 * 1024 * 1024)
    service_auth_token: SecretStr | None = None
    service_auth_previous_token: SecretStr | None = None

    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=120, allow_inf_nan=False)
    llm_max_output_tokens: int = Field(default=4096, ge=1, le=16_384)
    llm_temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)

    model_config = SettingsConfigDict(
        env_file=os.environ.get("QUESTIONNAIRE_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        """
        Проверяет секрет без привязки к префиксу конкретного провайдера.
        """
        if value is None:
            return None

        secret = value.get_secret_value()
        if not 1 <= len(secret) <= 4096:
            raise ValueError("Ключ LLM должен содержать от 1 до 4096 символов")
        if secret != secret.strip() or any(character.isspace() for character in secret):
            raise ValueError("Ключ LLM не должен содержать пробельные символы")
        if not secret.isprintable():
            raise ValueError("Ключ LLM не должен содержать управляющие символы")
        return value

    @field_validator("service_auth_token", "service_auth_previous_token")
    @classmethod
    def validate_service_auth_token(cls, value: SecretStr | None) -> SecretStr | None:
        """
        Проверяет длину и алфавит непрозрачного Bearer-токена backend-сервиса.
        """
        if value is None:
            return None

        secret = value.get_secret_value()
        if not 32 <= len(secret) <= 512:
            raise ValueError("Сервисный токен должен содержать от 32 до 512 символов")
        if not secret.isascii() or any(
            character not in _SERVICE_TOKEN_CHARACTERS for character in secret
        ):
            raise ValueError("Сервисный токен содержит недопустимые символы")
        return value

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, value: str | None) -> str | None:
        """
        Проверяет непустой идентификатор модели без скрытой нормализации.
        """
        if value is None:
            return None
        if not 1 <= len(value) <= 255:
            raise ValueError("Идентификатор модели должен содержать от 1 до 255 символов")
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("Идентификатор модели не должен содержать пробельные символы")
        if not value.isprintable():
            raise ValueError("Идентификатор модели не должен содержать управляющие символы")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str | None) -> str | None:
        """
        Проверяет безопасный базовый HTTP-адрес OpenAI-совместимого endpoint.
        """
        if value is None:
            return None
        if not 1 <= len(value) <= 2048 or value != value.strip():
            raise ValueError("Базовый адрес LLM задан некорректно")
        if (
            any(character.isspace() for character in value)
            or not value.isprintable()
            or "\\" in value
        ):
            raise ValueError("Базовый адрес LLM содержит недопустимые символы")

        try:
            parsed_url = urlsplit(value)
            parsed_url.port
        except ValueError:
            raise ValueError("Базовый адрес LLM задан некорректно") from None

        if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
            raise ValueError("Базовый адрес LLM должен быть абсолютным HTTP(S) URL")
        if "%" in parsed_url.netloc or parsed_url.port == 0:
            raise ValueError("Базовый адрес LLM содержит недопустимый host или port")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("Базовый адрес LLM не должен содержать учётные данные")
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("Базовый адрес LLM не должен содержать query или fragment")
        return value

    @field_validator("llm_timeout_seconds", "llm_temperature", mode="before")
    @classmethod
    def reject_boolean_float_setting(cls, value: object) -> object:
        """
        Не допускает преобразование логического значения в вещественную настройку.
        """
        if isinstance(value, bool):
            raise ValueError("Вещественная настройка LLM не может быть логическим значением")
        return value

    @field_validator("api_max_request_bytes", "llm_max_output_tokens", mode="before")
    @classmethod
    def validate_integer_limit_type(cls, value: object) -> object:
        """
        Принимает целый лимит или его десятичную запись из окружения.
        """
        if isinstance(value, bool):
            raise ValueError("Числовой лимит не может быть логическим значением")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isascii() and value.isdigit():
            return value
        raise ValueError("Числовой лимит должен быть целым числом")

    @model_validator(mode="after")
    def validate_llm_configuration(self) -> Self:
        """
        Проверяет целостность конфигурации LLM и защиту рабочего endpoint.
        """
        has_any_connection_value = any(
            value is not None for value in (self.llm_api_key, self.llm_model, self.llm_base_url)
        )
        if has_any_connection_value and (self.llm_api_key is None or self.llm_model is None):
            raise ValueError("LLM_API_KEY и LLM_MODEL должны быть заданы вместе")
        if not has_any_connection_value and self.app_env != "local":
            raise ValueError("Для окружений dev, test и prod требуется конфигурация LLM")
        if self.llm_configured and self.service_auth_token is None:
            raise ValueError("Для включённой обработки ходов требуется SERVICE_AUTH_TOKEN")
        if self.service_auth_previous_token is not None and self.service_auth_token is None:
            raise ValueError("SERVICE_AUTH_PREVIOUS_TOKEN допустим только вместе с текущим токеном")
        if (
            self.service_auth_token is not None
            and self.service_auth_previous_token is not None
            and compare_digest(
                self.service_auth_token.get_secret_value(),
                self.service_auth_previous_token.get_secret_value(),
            )
        ):
            raise ValueError("Текущий и предыдущий сервисные токены должны различаться")
        if self.llm_base_url is not None:
            parsed_url = urlsplit(self.llm_base_url)
            if parsed_url.scheme == "http" and not (
                self.app_env in {"local", "test"} and _is_loopback_host(parsed_url.hostname)
            ):
                raise ValueError(
                    "Обычный HTTP для LLM разрешён только на loopback в local или test"
                )
        return self

    @property
    def llm_configured(self) -> bool:
        """
        Показывает, достаточно ли настроек для создания LLM-клиента.
        """
        return self.llm_api_key is not None and self.llm_model is not None

    @property
    def service_auth_enabled(self) -> bool:
        """
        Показывает, защищён ли бизнес-маршрут сервисным Bearer-токеном.
        """
        return self.service_auth_token is not None


@lru_cache
def get_settings() -> Settings:
    """
    Возвращает единый проверенный набор настроек для процесса.
    """
    return Settings()


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

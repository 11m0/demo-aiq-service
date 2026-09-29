"""
Определяет безопасные предметные ошибки backend-приложения.
"""


class SessionNotFoundError(LookupError):
    """
    Сообщает об отсутствующей или истёкшей сессии.
    """


class SessionCapacityError(RuntimeError):
    """
    Сообщает о достижении настроенного лимита активных сессий.
    """


class StateVersionConflictError(RuntimeError):
    """
    Сообщает, что клиент отправил команду для устаревшей версии состояния.
    """

    def __init__(self, current_version: int) -> None:
        """
        Сохраняет актуальную версию для безопасного ответа клиенту.
        """
        self.current_version = current_version
        super().__init__("Версия состояния устарела")


class MessageIdConflictError(RuntimeError):
    """
    Сообщает о повторном использовании messageId для другой команды.
    """


class SessionCompletedError(RuntimeError):
    """
    Сообщает о попытке отправить реплику в завершённую сессию.
    """


class DialogModeNotSelectedError(RuntimeError):
    """
    Сообщает, что содержательная реплика пришла до выбора сценария.
    """


class DialogModeAlreadySelectedError(RuntimeError):
    """
    Сообщает о попытке повторно изменить уже выбранный сценарий.
    """


class MLTransportError(RuntimeError):
    """
    Сообщает о недоступности ML-сервиса без раскрытия сетевых деталей.
    """


class MLProtocolError(RuntimeError):
    """
    Сообщает о некорректном HTTP- или предметном ответе ML-сервиса.
    """


class MLTurnFailureError(RuntimeError):
    """
    Передаёт контролируемый отказ ML-сервиса без изменения сессии.
    """

    def __init__(self, code: str, message: str, retryable: bool) -> None:
        """
        Сохраняет безопасные поля предметной ошибки.
        """
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        super().__init__(message)

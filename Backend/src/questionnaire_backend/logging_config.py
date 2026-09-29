"""
Настраивает журналирование служебных событий backend-приложения.
"""

import logging


def configure_logging() -> None:
    """
    Показывает INFO-события backend в консоли без изменения логов библиотек.
    """
    backend_logger = logging.getLogger("questionnaire_backend")
    backend_logger.setLevel(logging.INFO)
    if backend_logger.hasHandlers():
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    backend_logger.addHandler(handler)
    backend_logger.propagate = False

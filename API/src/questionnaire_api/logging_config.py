"""
Настраивает стандартное журналирование HTTP-приложения.
"""

import logging


def configure_logging(level: str) -> None:
    """
    Включает единый формат журналов с указанным уровнем подробности.
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

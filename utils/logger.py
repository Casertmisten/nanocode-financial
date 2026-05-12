"""基于全局配置的日志记录器"""

import logging


class BaseLogger:
    """基于全局配置的日志记录器"""

    _base_logger = logging.getLogger("BaseLogger")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _base_logger.addHandler(stream_handler)
    _base_logger.setLevel(logging.INFO)

    @classmethod
    def getLogger(cls, name: str) -> logging.Logger:  # noqa: N802
        """独立名称的日志记录器"""
        return cls._base_logger.getChild(name)

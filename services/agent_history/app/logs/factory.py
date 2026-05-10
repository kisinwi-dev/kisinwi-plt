import logging
from logging.handlers import RotatingFileHandler
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import LOG_LEVEL, LOG_DATEFMT, LOG_JSON_PATH


# ----------- GET LOGGER -----------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # Docker (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(TextLogFormatter(datefmt=LOG_DATEFMT))
    logger.addHandler(console_handler)

    # JSON file
    json_path = Path(LOG_JSON_PATH)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = 10 * 1024 * 1024
    json_handler = RotatingFileHandler(
        json_path, maxBytes=max_bytes, backupCount=5, encoding="utf-8"
    )
    json_handler.setLevel(LOG_LEVEL)
    json_handler.setFormatter(JSONLogFormatter())
    logger.addHandler(json_handler)

    logger.propagate = False
    return logger


# ----------- FORMAT -----------


class TextLogFormatter(logging.Formatter):
    LEVEL_STYLES = {
        logging.DEBUG:    "🔎 DEBUG ",
        logging.INFO:     "💙 INFO  ",
        logging.WARNING:  "💛 WARN  ",
        logging.ERROR:    "💔 ERROR ",
        logging.CRITICAL: "🔥 CRIT  ",
    }

    def __init__(self, datefmt: str | None = None):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt=datefmt,
        )

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self.LEVEL_STYLES.get(record.levelno, record.levelname)
        return super().format(record)


class JSONLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # extra
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process"
            ):
                continue
            log[key] = value

        # exception
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        return json.dumps(log, ensure_ascii=False)

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock


_LOG_FILE_PATH: Path | None = None
_LOG_FILE_LOCK = Lock()


def _get_log_file_path() -> Path:
    """
    Get one log file path for this run.
    """
    global _LOG_FILE_PATH

    if _LOG_FILE_PATH is not None:
        return _LOG_FILE_PATH

    with _LOG_FILE_LOCK:
        if _LOG_FILE_PATH is not None:
            return _LOG_FILE_PATH

        project_root = Path(__file__).resolve().parent.parent
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _LOG_FILE_PATH = logs_dir / f"selfplay_{timestamp}.txt"
        return _LOG_FILE_PATH


def log_line(message: str) -> None:
    """
    Write one plain text line.
    """
    file_path = _get_log_file_path()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    line = f"[{now}] {message}\n"

    with _LOG_FILE_LOCK:
        with file_path.open("a", encoding="utf-8") as file:
            file.write(line)


def log_event(tag: str, payload: dict) -> None:
    """
    Write one structured log line: tag + JSON.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    log_line(f"{tag} {text}")


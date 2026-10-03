from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class JsonlRequestLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, record: dict):
        value = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with self.lock, self.path.open("a") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

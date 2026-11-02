from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent.parent
SESSION_FILE = BASE_DIR / "session.json"
_lock = Lock()


def _default():
    return {"dataset": None}


class Session:
    def load(self):
        with _lock:
            try:
                return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                data = _default()
                self._write(data)
                return data

    def _write(self, data):
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, SESSION_FILE)

    def save(self, data):
        with _lock:
            self._write(data)

    def set_dataset(self, dataset: str):
        data = self.load()
        data["dataset"] = dataset
        self.save(data)

    def get_dataset(self):
        return self.load().get("dataset")


session = Session()

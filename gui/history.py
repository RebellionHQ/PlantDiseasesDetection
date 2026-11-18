import base64
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("PLANT_APP_DATA", Path.home() / ".plant_doctor"))
MAX_ENTRIES = 50


class History:
    def __init__(self, directory: str | Path = DATA_DIR):
        self.directory = Path(directory)
        self.path = self.directory / "history.json"
        self.entries = self._read()

    def _read(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)][:MAX_ENTRIES]

    def _write(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.entries, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def add(self, result: dict, image: bytes, source: str) -> dict:
        entry = {
            "id": uuid.uuid4().hex,
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "image": base64.b64encode(image).decode("ascii"),
            "plant": result["plant"],
            "disease": result["disease"],
            "confidence": result["confidence"],
            "top": result.get("top", []),
        }
        self.entries.insert(0, entry)
        del self.entries[MAX_ENTRIES:]
        self._write()
        return entry

    def clear(self) -> None:
        self.entries = []
        self._write()

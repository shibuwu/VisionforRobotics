import json
from pathlib import Path

class JsonlLogger:
    def __init__(self, out_dir: Path, filename: str = "history.jsonl"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / filename

    def log(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

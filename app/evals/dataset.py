from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_eval_dataset(dataset_path: Path | None = None) -> list[dict[str, Any]]:
    path = dataset_path or Path("app/evals/test_questions.yaml")
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return list(payload)

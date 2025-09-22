import csv
import json
import os
from typing import List, Dict, Any
from app.settings import settings

def write_json(rows: List[Dict[str, Any]], fname: str = "results.json") -> str:
    path = os.path.join(settings.tmp_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path

def write_csv(rows: List[Dict[str, Any]], fname: str = "results.csv") -> str:
    if not rows:
        raise ValueError("No rows to write")
    path = os.path.join(settings.tmp_dir, fname)
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path

def write_text(text: str, fname: str) -> str:
    path = os.path.join(settings.tmp_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
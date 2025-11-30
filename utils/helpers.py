import re
from typing import Any, Dict, List, Tuple
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

# FIX: read PROMPT_SYSTEM from config
SYSTEM_PROMPT = config.get("PROMPT_SYSTEM", """
Bạn là công cụ trích xuất thông tin từ file PDF (hóa đơn/biên lai/chứng từ).
Nhiệm vụ:
- Đọc toàn bộ nội dung PDF
- Sửa lỗi OCR (dấu tiếng Việt, chính tả)
- Ánh xạ dữ liệu vào danh sách JSON, mỗi phần tử là một entry (có thể nhiều entry).
""")

def build_prompt(fields):
    bullets = "\n".join([f'- "{f}"' for f in fields])
    example = ",\n".join([f'      "{f}": null' for f in fields])
    return SYSTEM_PROMPT.format(
        fields_bullets=bullets,
        fields_example=example,
    )

def clean_json_output(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return text
    # code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if m: 
        return m.group(1)
    # greedy braces
    m = re.search(r"\{[\s\S]*\}", text)
    if m: 
        return m.group(0)
    return text

# ---- parsing helpers for metadata + table rows ----
def _records_from_table_struct(table: Any) -> List[Dict[str, Any]]:
    if isinstance(table, dict):
        columns = table.get("columns", [])
        rows = table.get("rows", [])
        if columns and rows:
            return [dict(zip(columns, row)) for row in rows]
    if isinstance(table, list) and table:
        columns = table[0]
        rows = table[1:]
        if isinstance(columns, list) and rows:
            return [dict(zip(columns, row)) for row in rows]
    return []

def split_metadata_and_rows(data: Any) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    metadata: Dict[str, Any] | None = None
    rows: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        metadata_candidate = data.get("metadata") or data.get("header")
        if isinstance(metadata_candidate, list):
            metadata_candidate = metadata_candidate[0] if metadata_candidate else None
        if isinstance(metadata_candidate, dict):
            metadata = metadata_candidate

        table_candidate = data.get("table_rows") or data.get("table") or data.get("rows")
        rows = _records_from_table_struct(table_candidate) if table_candidate is not None else []
        if not rows and isinstance(table_candidate, list) and all(isinstance(r, dict) for r in table_candidate):
            rows = table_candidate

    elif isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]

    return metadata, rows

def normalize_fields(record: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {field: record.get(field, None) for field in fields}

def merge_metadata_rows(metadata: Dict[str, Any] | None, rows: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    if rows:
        for row in rows:
            merged.append({
                field: row.get(field) if row.get(field) not in (None, "") else (metadata.get(field) if metadata else None)
                for field in fields
            })
    elif metadata:
        merged = [metadata]
    return merged

def parse_model_payload(payload: Any, fields: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, List[Dict[str, Any]]]:
    metadata_raw, rows_raw = split_metadata_and_rows(payload)
    metadata_norm = normalize_fields(metadata_raw, fields) if metadata_raw else None
    rows_norm = [normalize_fields(r, fields) for r in rows_raw] if rows_raw else []
    merged = merge_metadata_rows(metadata_norm, rows_norm, fields)
    return merged, metadata_norm, rows_norm

def get_fields_from_table(df_like):
    """Accepts a Gradio Dataframe value; returns a flat list of field names."""
    if df_like is None:
        return []
    try:
        if hasattr(df_like, "values"):  # pandas DataFrame or numpy-like
            if getattr(df_like, "empty", False):
                return []
            rows = df_like.values.tolist()
        else:
            rows = df_like if isinstance(df_like, list) else []
        return [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]
    except Exception:
        # fallback: assume iterable of rows
        try:
            return [str(r[0]).strip() for r in df_like if r and str(r[0]).strip()]
        except Exception:
            return []

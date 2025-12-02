import json
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
METADATA_FIELDS: List[str] = config.get("METADATA_FIELDS", [])

def build_prompt(fields):
    def bullets(fs: List[str]) -> str:
        return "\n".join([f'- "{f}"' for f in fs])

    def example(fs: List[str]) -> str:
        return ",\n".join([f'      "{f}": null' for f in fs])

    metadata_fs = [f for f in METADATA_FIELDS if f in fields]
    table_fs = [f for f in fields if f not in metadata_fs]

    return SYSTEM_PROMPT.format(
        fields_bullets=bullets(fields),
        fields_example=example(fields),
        metadata_fields_bullets=bullets(metadata_fs),
        metadata_fields_example=example(metadata_fs),
        table_fields_bullets=bullets(table_fs),
        table_fields_example=example(table_fs),
    )

def clean_json_output(text: str) -> str:
    """
    Best-effort cleaning:
    - If already valid JSON, return as-is.
    - If two JSON objects (JSONL style), wrap them into an array string.
    - If JSONL contains `null` and an object, wrap both into an array.
    - Otherwise fall back to the broad brace/array extraction.
    """
    text = text.strip()

    def _is_json(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    if _is_json(text):
        return text

    # code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if m:
        candidate = m.group(1).strip()
        if _is_json(candidate):
            return candidate
        text = candidate  # continue parsing this block

    # Detect two top-level objects separated by whitespace/newline
    m_pair = re.match(r"(\{[\s\S]*?\})\s*(\{[\s\S]*\})\s*$", text, re.DOTALL)
    if m_pair:
        candidate = "[" + ",".join([m_pair.group(1), m_pair.group(2)]) + "]"
        if _is_json(candidate):
            return candidate

    # Detect multiple JSON values (JSONL-style)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        candidate = "[" + ",".join(lines) + "]"
        if _is_json(candidate):
            return candidate

    # Detect multiple JSON objects (brace-style) – only if they look like siblings
    objects = re.findall(r"\{[\s\S]*?\}", text)
    if len(objects) > 1:
        return "[" + ",".join(objects) + "]"
    if len(objects) == 1 and _is_json(objects[0]):
        return objects[0]

    # Array fallback
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match and _is_json(array_match.group(0)):
        return array_match.group(0)

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
        if len(data) == 2 and (isinstance(data[1], dict) or isinstance(data[1], list)):
            metadata_candidate = data[0] if isinstance(data[0], dict) else None
            table_candidate = data[1]
            metadata = metadata_candidate
            rows = _records_from_table_struct(table_candidate) if table_candidate is not None else []
            if not rows and isinstance(table_candidate, list) and all(isinstance(r, dict) for r in table_candidate):
                rows = table_candidate
            if not rows and isinstance(table_candidate, dict):
                table_rows = table_candidate.get("table_rows") or table_candidate.get("rows")
                rows = _records_from_table_struct(table_rows) if table_rows is not None else []
                if not rows and isinstance(table_rows, list) and all(isinstance(r, dict) for r in table_rows):
                    rows = table_rows
        else:
            rows = [r for r in data if isinstance(r, dict)]

    return metadata, rows

def normalize_fields(record: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {field: record.get(field, None) for field in fields}

def merge_metadata_rows(
    metadata: Dict[str, Any] | None,
    rows: List[Dict[str, Any]],
    metadata_fields: List[str],
    table_fields: List[str],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    full_fields = metadata_fields + [f for f in table_fields if f not in metadata_fields]
    if rows:
        for row in rows:
            merged.append({
                field: (
                    row.get(field)
                    if field in table_fields
                    else (metadata.get(field) if metadata else None)
                )
                for field in full_fields
            })
    elif metadata:
        merged = [{field: metadata.get(field) for field in full_fields}]
    return merged

def parse_model_payload(
    payload: Any,
    fields: List[str],
    metadata_fields: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, List[Dict[str, Any]]]:
    metadata_fields = [f for f in (metadata_fields or METADATA_FIELDS) if f in fields]
    table_fields = [f for f in fields if f not in metadata_fields]

    # Strict schema: metadata + table fields must exactly cover the DEFAULT_FIELDS order.
    if len(metadata_fields) + len(table_fields) != len(fields):
        raise ValueError("Metadata fields + table fields do not cover DEFAULT_FIELDS")

    metadata_raw, rows_raw = split_metadata_and_rows(payload)
    metadata_norm = normalize_fields(metadata_raw, metadata_fields) if metadata_raw else None
    rows_norm = [normalize_fields(r, table_fields) for r in rows_raw] if rows_raw else []
    merged = merge_metadata_rows(metadata_norm, rows_norm, metadata_fields, table_fields)
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

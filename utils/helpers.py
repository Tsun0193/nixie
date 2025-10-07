import re
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
    # Expecting a list of field names; you already used this in your main flow.
    # This constructs the system/user prompt content.
    fields_txt = ", ".join([str(f) for f in fields])
    return f"""{SYSTEM_PROMPT}

Yêu cầu trích xuất các trường sau, trả về JSON hợp lệ (1 hoặc nhiều objects):
[{fields_txt}]
"""

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

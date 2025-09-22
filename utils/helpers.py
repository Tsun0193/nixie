import re
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SYSTEM_PROMPT = config.get("SYSTEM_PROMPT", """
    Bạn là công cụ trích xuất thông tin từ file PDF (hóa đơn/biên lai/chứng từ).
    Nhiệm vụ:
    - Đọc toàn bộ nội dung PDF
    - Sửa lỗi OCR (dấu tiếng Việt, chính tả)
    - Ánh xạ dữ liệu vào danh sách JSON, mỗi phần tử là một entry (có thể nhiều entry).
    [
    {{
    {fields_text}
    }} ,
    ...
    ]
    Chỉ trả về JSON hợp lệ, không kèm thêm giải thích.
""")                 

def build_prompt(custom_fields):
    fields_text = "\n".join([f'    \"{f}\": ,' for f in custom_fields])
    return SYSTEM_PROMPT.format(fields_text=fields_text)

def clean_json_output(output: str) -> str:
    output = re.sub(r"^```[a-zA-Z]*\n?", "", output.strip())
    output = re.sub(r"\n?```$", "", output.strip())
    return output.strip()

def get_fields_from_table(table):
    if table is None:
        return []
    if hasattr(table, "values"):
        table = table.values.tolist()
    return [row[0].strip() for row in table if row and row[0] and str(row[0]).strip()]

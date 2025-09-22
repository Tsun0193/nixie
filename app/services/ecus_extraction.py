from typing import Dict, List, Any, Tuple
import json
from app.clients.openai_client import get_openai_client
from app.settings import settings
from utils.helpers import clean_json_output  # your existing helper

# Header fields we’ll fill (matches template placeholders)
ECUS_HEADER_FIELDS: List[str] = [
    "SoToKhai", "NgayKhaiBao", "LoaiHinh", "HaiQuan",
    "ExporterName", "ExporterAddress", "ExporterCountryCode",
    "ImporterName", "ImporterAddress", "ImporterTaxCode",
    "PhuongTien", "BillOfLading", "PortOfEntry",
    "ImportDuty", "VAT", "TotalTax",
]

# Per-item goods fields
ECUS_GOODS_FIELDS: List[str] = [
    "Description", "HSCode", "OriginCountry", "Quantity", "Unit", "CIFValue", "Currency"
]

def _build_prompt(header_keys: List[str], goods_keys: List[str]) -> str:
    # Keep it strict JSON, like your main flow
    return (
        "Bạn là công cụ trích xuất thông tin từ PDF tờ khai/biên lai.\n"
        "Đầu vào là 1 file PDF. Nhiệm vụ:\n"
        "1) Sửa lỗi OCR hiển nhiên (đặc biệt dấu tiếng Việt), giữ nguyên số/ký hiệu.\n"
        "2) Chuẩn hóa ngày (ưu tiên yyyy-mm-dd hoặc mm/dd/yyyy), giữ nguyên khi không chắc.\n"
        "3) Trả về JSON **hợp lệ** đúng cấu trúc sau (không kèm giải thích):\n\n"
        "{\n"
        '  "header": {\n' +
        "".join([f'    "{k}": "",\n' for k in header_keys])[:-2] + "\n"
        "  },\n"
        '  "goods": [\n'
        "    {\n" +
        "".join([f'      "{k}": "",\n' for k in goods_keys])[:-2] + "\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "- Nếu không có hàng hóa, để goods = [].\n"
        "- Chỉ trả về JSON hợp lệ."
    )

def extract_ecus_from_pdf(pdf_path: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    client = get_openai_client()
    prompt = _build_prompt(ECUS_HEADER_FIELDS, ECUS_GOODS_FIELDS)

    uploaded = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
    file_id = uploaded.id
    try:
        resp = client.responses.create(
            model=settings.model_name,  # same as main flow (default gpt-4.1-mini)
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id}
                ]
            }]
        )
        raw = resp.output_text.strip()
        cleaned = clean_json_output(raw)

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Model did not return a JSON object")

        header = data.get("header", {}) or {}
        goods = data.get("goods", []) or []

        # Coerce to known keys only (avoid surprises)
        header_coerced = {k: str(header.get(k, "")) for k in ECUS_HEADER_FIELDS}
        goods_coerced = [
            {k: str(row.get(k, "")) for k in ECUS_GOODS_FIELDS}
            for row in goods if isinstance(row, dict)
        ]
        return header_coerced, goods_coerced
    finally:
        client.files.delete(file_id)
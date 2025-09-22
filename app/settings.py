from pathlib import Path
from pydantic import BaseModel
import os
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    config_path: Path = BASE_DIR / "config.yaml"
    template_xml_path: Path = BASE_DIR / "template" / "template.xml"
    tmp_dir: Path = Path(os.getenv("TMP_DIR", "/tmp"))

    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

settings = Settings()

# Load YAML config (required)
with open(settings.config_path, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

DEFAULT_FIELDS = CONFIG.get("DEFAULT_FIELDS", [
    "Số phiếu", "Ngày chứng từ(mm/dd/yyyy)", "Ngày xuất(mm/dd/yyyy)",
    "Mã FG", "Đvt", "Số lượng(Thùng)", "Số lượng(Pcs)",
    "Kho nhận", "Địa chỉ", "Mã AR"
])

# Load ECUS template (required)
with open(settings.template_xml_path, "r", encoding="utf-8") as f:
    TEMPLATE_XML = f.read()

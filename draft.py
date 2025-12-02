from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import csv
import os
import re
from typing import List

from utils.helpers import parse_model_payload, clean_json_output, METADATA_FIELDS

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc

# Avoid flash_attn binary issues by using SDPA attention instead of flash_attention_2.
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # fall back to PyTorch SDPA (works without flash-attn / GPU)
    device_map="cuda",  # will place on GPU if visible, else CPU
    cache_dir=".cache"
)
model.eval()

print(f"Number of parameters: {model.num_parameters()/1e9:.2f}B")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Load expected fields and prompt from config for guidance and consistent CSV headers.
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
default_fields: List[str] = config.get("DEFAULT_FIELDS", [])
metadata_fields: List[str] = config.get("METADATA_FIELDS", METADATA_FIELDS)

def build_prompt(template: str, fields: List[str], metadata_fields: List[str]) -> str:
    def bullets(fs: List[str]) -> str:
        return "\n".join([f'- "{f}"' for f in fs])

    def example(fs: List[str]) -> str:
        return ",\n".join([f'      "{f}": null' for f in fs])

    metadata_fs = [f for f in metadata_fields if f in fields]
    table_fs = [f for f in fields if f not in metadata_fs]

    return template.format(
        fields_bullets=bullets(fields),
        fields_example=example(fields),
        metadata_fields_bullets=bullets(metadata_fs),
        metadata_fields_example=example(metadata_fs),
        table_fields_bullets=bullets(table_fs),
        table_fields_example=example(table_fs),
    )

prompt_system = build_prompt(
    config.get("PROMPT_SYSTEM")
    or (
        "Extract the following fields into a JSON array of objects using these exact keys: "
        + ", ".join(default_fields)
        if default_fields
        else "Extract the table from the provided image and parse it into JSON format."
    ),
    default_fields,
    metadata_fields,
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": prompt_system},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "data/KO_img/PGH Ban Doan Thi Thuy 1N2.01_page_001.png",
            },
            {
                "type": "text", 
                "text": "Return only the JSON as instructed (no explanations)."
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)

raw_text = output_text[0] if output_text else ""
clean_text = re.sub(r"^```(?:json)?\n?", "", raw_text).rstrip("`").strip()

cleaned = clean_json_output(clean_text)
try:
    data = json.loads(cleaned)
except Exception as exc:
    raise ValueError(f"Could not parse JSON from model output: {exc}") from exc

merged_records, metadata_norm, rows_norm = parse_model_payload(
    data, default_fields, metadata_fields=metadata_fields
)

os.makedirs("output", exist_ok=True)
csv_path = os.path.join("output", "extracted_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    headers = default_fields if default_fields else (list(merged_records[0].keys()) if merged_records else [])
    writer.writerow(headers)
    for record in merged_records:
        writer.writerow([record.get(field, "") for field in headers])

with open(os.path.join("output", "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata_norm if metadata_norm else None, f, ensure_ascii=False, indent=2)

with open(os.path.join("output", "table_rows.json"), "w", encoding="utf-8") as f:
    json.dump(rows_norm, f, ensure_ascii=False, indent=2)

print(f"Table saved to {csv_path}")

from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import csv
import os
import re
from typing import Any, Dict, List, Tuple

from utils.helpers import parse_model_payload

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required. Install with `pip install -r requirements.txt`.") from exc

# Avoid flash_attn binary issues by using SDPA attention instead of flash_attention_2.
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # fall back to PyTorch SDPA (works without flash-attn / GPU)
    device_map="cuda:1",  # will place on GPU if visible, else CPU
    cache_dir=".cache"
)
model.eval()

print(f"Number of parameters: {model.num_parameters()/1e9:.2f}B")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Load expected fields and prompt from config for guidance and consistent CSV headers.
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
default_fields: List[str] = config.get("DEFAULT_FIELDS", [])

def build_prompt(template: str, fields: List[str]) -> str:
    bullets = "\n".join([f'- "{f}"' for f in fields])
    example = ",\n".join([f'      "{f}": null' for f in fields])
    return template.format(
        fields_bullets=bullets,
        fields_example=example,
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

def extract_json(text: str):
    """Best-effort JSON extraction from possibly noisy model output."""
    text = text.strip()

    def try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except Exception:
            return None

    parsed = try_parse(text)
    if parsed is not None:
        return parsed

    array_match = re.search(r"\[[\s\S]*?\]", text)
    if array_match:
        parsed = try_parse(array_match.group(0))
        if parsed is not None:
            return parsed

    objects = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        candidate = try_parse(match.group(0))
        if candidate is not None:
            objects.append(candidate)
    if objects:
        return objects

    raise ValueError("Could not parse JSON from model output")

data = extract_json(clean_text)

merged_records, metadata_norm, rows_norm = parse_model_payload(data, default_fields)

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

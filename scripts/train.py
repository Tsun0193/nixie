"""Fine-tune Qwen3-VL-8B-Instruct on the local JSON + image dataset.

This script expects `data/training_data.json` to contain entries with:
- source_file: path to the PDF (used to derive the page image name)
- page_number / total_pages
- metadata: list/dict with header fields
- table_rows: list of row dicts

Images are resolved from `--image-root` using the pattern:
    <image-root>/<pdf-stem>_page_<page_number:03d>.png
Adjust `--image-root` if your images live elsewhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

try:
    from trl import SFTConfig, SFTTrainer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "trl is required for this script. Install with `pip install trl` (or add to pyproject)."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required. Install with `pip install pyyaml`.") from exc

# Ensure project root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import build_prompt

LOGGER = logging.getLogger(__name__)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_answer(
    record: Dict[str, Any],
    metadata_fields: List[str],
    table_fields: List[str],
) -> str:
    """Build the two-line JSON target expected by the system prompt."""
    metadata_obj = record.get("metadata")
    if isinstance(metadata_obj, list):
        metadata_obj = metadata_obj[0] if metadata_obj else None
    if isinstance(metadata_obj, dict):
        metadata_obj = {field: metadata_obj.get(field) for field in metadata_fields}
    else:
        metadata_obj = None

    rows_norm = []
    for row in record.get("table_rows") or []:
        if isinstance(row, dict):
            rows_norm.append({field: row.get(field) for field in table_fields})
    table_obj = {"table_rows": rows_norm}

    # Keep formatting simple: one JSON object per line to mirror the prompt.
    return (
        json.dumps(metadata_obj, ensure_ascii=False)
        + "\n"
        + json.dumps(table_obj, ensure_ascii=False)
    )


def _resolve_image_path(image_root: Path, source_file: str, page_number: int) -> Optional[Path]:
    source_stem = Path(source_file.replace("\\", "/")).stem
    candidate = image_root / f"{source_stem}_page_{page_number:03d}.png"
    if candidate.exists():
        return candidate
    LOGGER.warning("Could not find image for %s page %s", source_file, page_number)
    return None


class QwenVLDataset(Dataset):
    """Dataset that pairs page images with the expected JSON response."""

    def __init__(
        self,
        data_path: Path,
        processor: AutoProcessor,
        system_prompt: str,
        metadata_fields: List[str],
        table_fields: List[str],
        image_root: Path,
        max_samples: Optional[int] = None,
    ) -> None:
        with data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.processor = processor
        self.system_prompt = system_prompt
        self.metadata_fields = metadata_fields
        self.table_fields = table_fields

        self.samples: List[Dict[str, Any]] = []
        for rec in raw:
            page_number = rec.get("page_number")
            if page_number is None:
                continue
            image_path = _resolve_image_path(image_root, rec.get("source_file", ""), int(page_number))
            if not image_path:
                continue
            self.samples.append(
                {
                    "image_path": image_path,
                    "page_number": int(page_number),
                    "total_pages": rec.get("total_pages"),
                    "source_file": rec.get("source_file"),
                    "answer": _format_answer(rec, metadata_fields, table_fields),
                }
            )
            if max_samples and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError(f"No usable samples loaded from {data_path}")
        LOGGER.info("Loaded %s samples (max_samples=%s).", len(self.samples), max_samples)
        self.column_names = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getattr__(self, name: str):
        if name == "column_names":
            return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
        raise AttributeError(name)

    # Minimal map implementation for TRL utilities that expect datasets.Dataset.
    def map(self, function, batched: bool = False, **kwargs):
        if not self.samples:
            return self
        if batched:
            keys = list(self.samples[0].keys())
            columns = {k: [s[k] for s in self.samples] for k in keys}
            new_columns = function(columns)
            # Rebuild samples from returned columns (assume same length).
            length = len(next(iter(new_columns.values()))) if new_columns else 0
            self.samples = [
                {k: new_columns.get(k, [None] * length)[i] for k in keys} for i in range(length)
            ]
        else:
            self.samples = [function(sample) for sample in self.samples]
        return self

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as im:
            image = im.convert("RGB")

        user_text = "\n".join(
            [
                f"Extract metadata + table for page {sample['page_number']}/{sample.get('total_pages') or '?'}",
                "Return exactly two JSON lines as described in the system prompt.",
            ]
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]

        # Build conversation strings for loss-masking.
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.processor(
            text=[full_text],
            images=[image],
            padding="longest",
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        pixel_values = model_inputs["pixel_values"][0]
        image_grid_thw = model_inputs.get("image_grid_thw")  # keep original shape for VLM

        labels = input_ids.clone()
        prompt_ids = self.processor.tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        labels[: prompt_ids.shape[0]] = -100  # mask system + user tokens

        if idx == 0:
            LOGGER.info(
                "Sample shapes - input_ids: %s, pixel_values: %s, image_grid_thw: %s",
                tuple(input_ids.shape),
                tuple(pixel_values.shape),
                None if image_grid_thw is None else tuple(image_grid_thw.shape),
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }


@dataclass
class VisionDataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([f.pop("pixel_values") for f in features])
        image_grid_thw_vals = [f.pop("image_grid_thw") for f in features]
        labels = [f.pop("labels") for f in features]

        token_features = {
            "input_ids": [f["input_ids"].tolist() if torch.is_tensor(f["input_ids"]) else f["input_ids"] for f in features],
            "attention_mask": [
                f["attention_mask"].tolist() if torch.is_tensor(f["attention_mask"]) else f["attention_mask"]
                for f in features
            ],
        }

        batch = self.processor.tokenizer.pad(token_features, padding=True, return_tensors="pt")

        label_inputs = {
            "input_ids": [lbl.tolist() if torch.is_tensor(lbl) else lbl for lbl in labels],
        }
        labels_padded = self.processor.tokenizer.pad(
            label_inputs,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels_padded[labels_padded == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels_padded
        batch["pixel_values"] = pixel_values
        if image_grid_thw_vals[0] is not None:
            tensors = []
            for v in image_grid_thw_vals:
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                if v.dim() == 1:
                    v = v.unsqueeze(0)  # ensure (num_images, 3)
                tensors.append(v)
            batch["image_grid_thw"] = torch.stack(tensors)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-8B-Instruct.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-file", default="data/training_data.json", type=Path)
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--image-root", default="data/LT_img", type=Path)
    parser.add_argument("--output-dir", default="output/qwen3-vl-8b-instruct-finetuned", type=Path)
    parser.add_argument("--cache-dir", default=".cache", type=Path)
    parser.add_argument("--num-epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-steps", default=10, type=int)
    parser.add_argument("--logging-steps", default=10, type=int)
    parser.add_argument("--save-steps", default=200, type=int)
    parser.add_argument("--save-total-limit", default=2, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--device-map",
        default="auto",
        type=str,
        help='Pass "auto" to spread across GPUs or "cpu"/"cuda:0" for a single device.',
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 when supported.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 when supported.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    set_seed(args.seed)

    config = _load_config(args.config)
    default_fields: List[str] = config.get("DEFAULT_FIELDS") or []
    metadata_fields: List[str] = config.get("METADATA_FIELDS") or []
    table_fields = [f for f in default_fields if f not in metadata_fields]
    if not default_fields:
        raise ValueError("DEFAULT_FIELDS is empty in config.yaml; cannot build prompts.")

    system_prompt = build_prompt(default_fields)

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    model.config.use_cache = False
    if model.config.pad_token_id is None and processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = QwenVLDataset(
        data_path=args.train_file,
        processor=processor,
        system_prompt=system_prompt,
        metadata_fields=metadata_fields,
        table_fields=table_fields,
        image_root=args.image_root,
        max_samples=args.max_samples,
    )

    collator = VisionDataCollator(processor=processor)

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_length=None,  # avoid TRL truncation/map over custom Dataset
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
        dataset_text_field=None,  # we return fully tokenized samples
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    LOGGER.info("Starting training with %s samples", len(dataset))
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    LOGGER.info("Training complete. Model + processor saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

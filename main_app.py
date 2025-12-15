import glob
import os
import json
import csv
import base64
import tempfile
import subprocess
from datetime import datetime
from typing import List, Dict, Any
import re

import yaml
# --- Gradio temp dir fix (must be set before importing gradio) ---
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_tmp")
os.environ["TMPDIR"] = os.environ["GRADIO_TEMP_DIR"]
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv()

SAMPLE_DIR = "data"
# Allow both PDFs and common image formats as samples
_sample_patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
SAMPLES = [
    {"label": os.path.basename(p), "path": p}
    for p in sorted(
        {path for pattern in _sample_patterns for path in glob.glob(os.path.join(SAMPLE_DIR, pattern))}
    )
]
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

# --- config / defaults ---
with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

DEFAULT_FIELDS: List[str] = _cfg.get("DEFAULT_FIELDS", [])
METADATA_FIELDS: List[str] = _cfg.get("METADATA_FIELDS", [])
PROMPT_SYSTEM: str = _cfg.get("PROMPT_SYSTEM", "")

# --- utils from your repo ---
from utils.helpers import clean_json_output, get_fields_from_table, parse_model_payload  # noqa: E402

# --- in-memory store ---
MAIN_ROWS: List[Dict[str, Any]] = []

# --- local model (same stack as draft.py) ---
_model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
    cache_dir=".cache",
)
_model.eval()
_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# --- helpers ---
def show_pdf(file) -> str:
    if file is None:
        return ""
    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    if ext == ".pdf":
        width = 700
        height = int(width * 1.414)
        return f'''
        <embed src="data:application/pdf;base64,{b64}"
               type="application/pdf"
               width="{width}px" height="{height}px"
               style="border:1px solid #ddd; margin:auto; display:block;" />
        '''
    return f'<img src="data:image;base64,{b64}" style="max-width:100%; height:auto; display:block; margin:auto;" />'

def write_json(rows: List[Dict[str, Any]]) -> str:
    path = os.path.join(tempfile.gettempdir(), "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path

def write_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("No rows")
    headers = list(rows[0].keys())
    path = os.path.join(tempfile.gettempdir(), "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    return path

# --- prompt builder ---
def _fields_as_bullets(fields: List[str]) -> str:
    return "\n".join([f"- \"{f}\"" for f in fields])

def _fields_as_json_example(fields: List[str]) -> str:
    return ",\n".join([f'      "{f}": null' for f in fields])

def _build_prompt(fields: List[str]) -> str:
    metadata_fields = [f for f in METADATA_FIELDS if f in fields]
    table_fields = [f for f in fields if f not in metadata_fields]
    return PROMPT_SYSTEM.format(
        fields_bullets=_fields_as_bullets(fields),
        fields_example=_fields_as_json_example(fields),
        metadata_fields_bullets=_fields_as_bullets(metadata_fields),
        metadata_fields_example=_fields_as_json_example(metadata_fields),
        table_fields_bullets=_fields_as_bullets(table_fields),
        table_fields_example=_fields_as_json_example(table_fields),
    )

# --- pdf helpers ---
def _pdf_to_image(pdf_path: str) -> str:
    """Convert the first page of a PDF to a temporary PNG."""
    tmp_dir = tempfile.mkdtemp(prefix="pdf_preview_")
    out_prefix = os.path.join(tmp_dir, "page")
    cmd = [
        "pdftoppm",
        "-png",
        "-singlefile",
        "-f",
        "1",
        "-l",
        "1",
        pdf_path,
        out_prefix,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("pdftoppm is required to process PDF files") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"Failed to convert PDF: {stderr}") from exc
    return f"{out_prefix}.png"

def _log_model_output(raw_text: str, cleaned_text: str) -> str:
    """Persist raw/cleaned model output for debugging."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(LOG_DIR, f"model_output_{ts}.json")
    payload = {
        "timestamp": ts,
        "raw": raw_text,
        "cleaned": cleaned_text,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Do not block extraction flow on logging issues
        return ""
    return path

def _autoclose_json(text: str) -> str:
    """Best-effort fix: balance brackets/braces and strip trailing commas."""
    fixed = re.sub(r",\s*([\]}])", r"\1", text.strip())
    def _balance(s: str, open_ch: str, close_ch: str) -> str:
        diff = s.count(open_ch) - s.count(close_ch)
        if diff > 0:
            s += close_ch * diff
        return s
    fixed = _balance(fixed, "{", "}")
    fixed = _balance(fixed, "[", "]")
    return fixed

# --- normalization rules ---
DATE_DOC = "Ngày chứng từ(mm/dd/yyyy)"
DATE_OUT = "Ngày xuất(mm/dd/yyyy)"
SO_PHIEU = "Số phiếu"

def _parse_mmddyyyy_or_none(s: str) -> str | None:
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    s_norm = re.sub(r"[.\- ]", "/", s)
    candidates = [s_norm]

    if re.fullmatch(r"\d{8}", re.sub(r"[^0-9]", "", s)):
        digits = re.sub(r"[^0-9]", "", s)
        candidates.append(f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}")

    fmts = ["%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y/%d/%m"]
    for cand in candidates:
        for fmt in fmts:
            try:
                dt = datetime.strptime(cand, fmt)
                return dt.strftime("%m/%d/%Y")
            except Exception:
                pass
    return None

def _normalize_rows(rows: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        item = {f: (r.get(f, None)) for f in fields}

        # "Số phiếu" → null if missing
        if not item.get(SO_PHIEU) or str(item.get(SO_PHIEU)).strip() == "":
            item[SO_PHIEU] = None

        # Normalize dates
        out_norm = _parse_mmddyyyy_or_none(item.get(DATE_OUT))
        doc_norm = _parse_mmddyyyy_or_none(item.get(DATE_DOC))

        # Prevent copy: if equal → null Ngày chứng từ
        if doc_norm and out_norm and doc_norm == out_norm:
            doc_norm = None

        item[DATE_OUT] = out_norm
        item[DATE_DOC] = doc_norm

        normalized.append(item)
    return normalized

# --- extraction (local model, image input) ---
def extract_from_pdf(pdf_path: str, fields: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, List[Dict[str, Any]]]:
    ext = os.path.splitext(pdf_path)[1].lower()
    img_path = _pdf_to_image(pdf_path) if ext == ".pdf" else pdf_path
    prompt = _build_prompt(fields)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Return only the JSON as instructed (no explanations)."},
            ],
        },
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model.device)
    generated_ids = _model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    raw_text = output_text[0] if output_text else ""
    raw_text = raw_text.strip()
    cleaned = clean_json_output(raw_text)
    log_path = _log_model_output(raw_text, cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback to YAML parser for slightly malformed JSON (e.g., trailing commas)
        try:
            data = yaml.safe_load(cleaned)
        except Exception:
            # Final attempt: auto-close brackets/braces
            fixed = _autoclose_json(cleaned)
            try:
                data = json.loads(fixed)
            except Exception as e2:
                snippet = fixed[:400].replace("\n", " ")
                raise ValueError(
                    f"Failed to parse model output as JSON/YAML after auto-fix. Error: {e2}. "
                    f"Snippet: {snippet}. Log: {log_path or 'n/a'}"
                ) from e2

    merged_rows, metadata_row, table_rows = parse_model_payload(data, fields)
    merged_rows = _normalize_rows(merged_rows, fields)
    return merged_rows, metadata_row, table_rows

# --- Gradio UI ---
def build_main_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 Multi-entry PDF Information Extraction (Main)")

        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(label="Upload Image/PDF", file_types=[".pdf", ".png", ".jpg", ".jpeg"], type="filepath")
                pdf_preview = gr.HTML(label="Preview")
            with gr.Column(scale=1):
                gr.Markdown("### Fields (edit/add/remove)")
                fields_table = gr.Dataframe(
                    headers=["Fields"],
                    datatype="str",
                    row_count=(len(DEFAULT_FIELDS), "dynamic"),
                    col_count=(1, "fixed"),
                    value=[[f] for f in DEFAULT_FIELDS],
                    interactive=True,
                    label="Fields"
                )

        pdf_file.upload(show_pdf, inputs=pdf_file, outputs=pdf_preview)

        if SAMPLES:
            gr.Markdown("### 📄 Or try sample files")

            sample_choices = gr.Dropdown(
                choices=[s["label"] for s in SAMPLES],
                label="Select a sample file to preview",
                interactive=True,
            )

            def load_sample(label):
                for s in SAMPLES:
                    if s["label"] == label:
                        return s["path"], show_pdf(s["path"])
                return None, ""

            sample_choices.change(
                load_sample,
                inputs=[sample_choices],
                outputs=[pdf_file, pdf_preview],
            )
        else:
            gr.Markdown("_No sample images or PDFs found in the `data/` folder._")

        extract_btn = gr.Button("Extract Info")
        status = gr.Label()

        demo_table = gr.Dataframe(
            headers=DEFAULT_FIELDS,
            datatype="str",
            row_count=(1, "dynamic"),
            col_count=(len(DEFAULT_FIELDS), "dynamic"),
            interactive=True,
            label="Extracted Entries (editable)"
        )
        metadata_view = gr.JSON(label="Metadata (header fields)")
        table_rows_view = gr.JSON(label="Raw table rows")

        def do_extract(file, table):
            if file is None:
                return [], {}, [], "⚠️ No file uploaded"
            fields = get_fields_from_table(table) or [r[0] for r in table if r and r[0]] or DEFAULT_FIELDS
            path = file if isinstance(file, str) else file.name
            try:
                merged_rows, metadata_row, table_rows = extract_from_pdf(path, fields)
            except Exception as e:
                return [], {}, [], f"⚠️ Extraction failed: {e}"
            if not merged_rows:
                return [], {}, [], "⚠️ No entries extracted."

            def _to_cell(v):
                return "" if v is None else str(v)

            grid = [[_to_cell(r.get(h, "")) for h in fields] for r in merged_rows]
            return (
                gr.update(value=grid, headers=fields, col_count=(len(fields), "dynamic")),
                metadata_row or {},
                table_rows or [],
                "✅ Extraction successful"
            )

        extract_btn.click(do_extract, inputs=[pdf_file, fields_table], outputs=[demo_table, metadata_view, table_rows_view, status])

        add_btn = gr.Button("Add to Results")
        session_status = gr.Label()

        def add_results(tbl):
            headers = tbl.headers if hasattr(tbl, "headers") else DEFAULT_FIELDS
            values = tbl.values.tolist() if hasattr(tbl, "values") else tbl
            added = 0
            for row in values:
                if any(str(x).strip() for x in row):
                    MAIN_ROWS.append(dict(zip(headers, row)))
                    added += 1
            return f"✅ Added {added} entries (total {len(MAIN_ROWS)})"

        add_btn.click(add_results, inputs=[demo_table], outputs=[session_status])

        with gr.Row():
            gr.Button("⬇️ Download JSON").click(None, js="() => { window.open('/download/json','_blank') }")
            gr.Button("⬇️ Download CSV").click(None, js="() => { window.open('/download/csv','_blank') }")

            def clear_all():
                MAIN_ROWS.clear()
                return "Cleared."
            gr.Button("Clear Session").click(clear_all, outputs=[session_status])

    return demo

# --- FastAPI app ---
app = FastAPI(title="Nixie Main App")

@app.get("/download/json")
def download_json():
    if not MAIN_ROWS:
        return {"error": "No results"}
    return FileResponse(write_json(MAIN_ROWS), filename="results.json", media_type="application/json")

@app.get("/download/csv")
def download_csv():
    if not MAIN_ROWS:
        return {"error": "No results"}
    return FileResponse(write_csv(MAIN_ROWS), filename="results.csv", media_type="text/csv")

# Mount Gradio last
ui = build_main_ui()
app = gr.mount_gradio_app(app, ui, path="/")

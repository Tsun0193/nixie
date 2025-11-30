import glob
import os
import json
import csv
import base64
import tempfile
from typing import List, Dict, Any
from datetime import datetime
import re

import yaml
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI

# --- Gradio temp dir fix ---
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

# --- load env / client ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
SAMPLE_DIR = "data"
SAMPLES = [
    {
        "label": os.path.basename(p),
        "path": p
    }
    for p in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.pdf")))
]

# --- config / defaults ---
with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

DEFAULT_FIELDS: List[str] = _cfg.get("DEFAULT_FIELDS", [])
PROMPT_SYSTEM: str = _cfg.get("PROMPT_SYSTEM", "")

# --- utils from your repo ---
from utils.helpers import clean_json_output, get_fields_from_table, parse_model_payload  # noqa: E402

# --- in-memory store ---
MAIN_ROWS: List[Dict[str, Any]] = []

# --- helpers ---
def show_pdf(file) -> str:
    if file is None:
        return ""
    path = file.name if hasattr(file, "name") else file
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    width = 700
    height = int(width * 1.414)
    return f'''
    <embed src="data:application/pdf;base64,{b64}"
           type="application/pdf"
           width="{width}px" height="{height}px"
           style="border:1px solid #ddd; margin:auto; display:block;" />
    '''

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
    return PROMPT_SYSTEM.format(
        fields_bullets=_fields_as_bullets(fields),
        fields_example=_fields_as_json_example(fields),
    )

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

# --- extraction ---
def extract_from_pdf(pdf_path: str, fields: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, List[Dict[str, Any]]]:
    prompt = _build_prompt(fields)
    uploaded = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
    fid = uploaded.id
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": fid}
                ]
            }]
        )
        raw = resp.output_text.strip()
        data = json.loads(clean_json_output(raw))
        merged_rows, metadata_row, table_rows = parse_model_payload(data, fields)

        # Apply business rules to merged rows
        merged_rows = _normalize_rows(merged_rows, fields)
        return merged_rows, metadata_row, table_rows
    finally:
        client.files.delete(fid)

# --- Gradio UI ---
def build_main_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 Multi-entry PDF Information Extraction (Main)")

        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                pdf_preview = gr.HTML(label="Preview PDF")
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
                label="Select a sample PDF to preview",
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
            gr.Markdown("_No sample PDFs found in the `data/` folder._")

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

        def do_extract(file, table):
            if file is None:
                return [], "⚠️ No file uploaded"
            fields = get_fields_from_table(table) or [r[0] for r in table if r and r[0]] or DEFAULT_FIELDS
            path = file if isinstance(file, str) else file.name
            try:
                merged_rows, _, _ = extract_from_pdf(path, fields)
            except Exception as e:
                return [], f"⚠️ Extraction failed: {e}"
            if not merged_rows:
                return [], "⚠️ No entries extracted."

            def _to_cell(v):
                return "" if v is None else str(v)

            grid = [[_to_cell(r.get(h, "")) for h in fields] for r in merged_rows]
            return gr.update(value=grid, headers=fields, col_count=(len(fields), "dynamic")), "✅ Extraction successful"

        extract_btn.click(do_extract, inputs=[pdf_file, fields_table], outputs=[demo_table, status])

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

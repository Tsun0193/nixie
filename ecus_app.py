# ecus_app.py
import os
import json
import csv
import base64
import tempfile
from typing import List, Dict, Tuple

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI

from utils.helpers import clean_json_output  # reuse your helper

# --- env / client ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# --- fields ---
ECUS_HEADER_FIELDS: List[str] = [
    "SoToKhai","NgayKhaiBao","LoaiHinh","HaiQuan",
    "ExporterName","ExporterAddress","ExporterCountryCode",
    "ImporterName","ImporterAddress","ImporterTaxCode",
    "PhuongTien","BillOfLading","PortOfEntry",
    "ImportDuty","VAT","TotalTax",
]
GOODS_FIELDS: List[str] = ["Description","HSCode","OriginCountry","Quantity","Unit","CIFValue","Currency"]

# --- template ---
with open("template/template.xml", "r", encoding="utf-8") as f:
    TEMPLATE_XML = f.read()

# --- in-memory state ---
ECUS_HEADER: Dict[str, str] = {k: "" for k in ECUS_HEADER_FIELDS}
ECUS_GOODS: List[Dict[str, str]] = []

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

def build_prompt(header_keys: List[str], goods_keys: List[str]) -> str:
    return (
        "Bạn là công cụ trích xuất thông tin từ PDF tờ khai/biên lai.\n"
        "Đầu vào là 1 file PDF. Nhiệm vụ:\n"
        "1) Sửa lỗi OCR hiển nhiên (dấu tiếng Việt), giữ nguyên số/ký hiệu.\n"
        "2) Chuẩn hóa ngày (yyyy-mm-dd hoặc mm/dd/yyyy nếu chắc chắn).\n"
        "3) Trả về JSON **hợp lệ** đúng cấu trúc sau (không giải thích thêm):\n\n"
        "{\n"
        '  "header": {\n' + "".join([f'    "{k}": "",\n' for k in header_keys])[:-2] + "\n  },\n"
        '  "goods": [\n'
        "    {\n" + "".join([f'      "{k}": "",\n' for k in goods_keys])[:-2] + "\n    }\n"
        "  ]\n"
        "}\n\n- Nếu không có hàng hóa, goods=[]."
    )

def extract_from_pdf(pdf_path: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    prompt = build_prompt(ECUS_HEADER_FIELDS, GOODS_FIELDS)
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
        if not isinstance(data, dict):
            raise ValueError("Model did not return an object")

        header = {k: str(data.get("header", {}).get(k, "")) for k in ECUS_HEADER_FIELDS}
        goods = [
            {k: str(row.get(k, "")) for k in GOODS_FIELDS}
            for row in (data.get("goods") or []) if isinstance(row, dict)
        ]
        return header, goods
    finally:
        client.files.delete(fid)

def build_goods_items(rows: List[Dict[str, str]]) -> str:
    chunks = []
    for r in rows:
        chunks.append(f"""        <Item>
            <Description>{r.get('Description','')}</Description>
            <HSCode>{r.get('HSCode','')}</HSCode>
            <OriginCountry>{r.get('OriginCountry','')}</OriginCountry>
            <Quantity>{r.get('Quantity','')}</Quantity>
            <Unit>{r.get('Unit','')}</Unit>
            <CIFValue>{r.get('CIFValue','')}</CIFValue>
            <Currency>{r.get('Currency','')}</Currency>
        </Item>""")
    return "\n".join(chunks)

def fill_template(header: Dict[str, str], goods_rows: List[Dict[str, str]]) -> str:
    mapping = {**header, "GoodsList": build_goods_items(goods_rows)}
    return TEMPLATE_XML.format(**mapping)

def write_text_file(text: str, name: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def write_csv_file(header: Dict[str, str], goods_rows: List[Dict[str, str]]) -> str:
    rows = [{**header, **g} for g in goods_rows]
    headers = list(header.keys()) + GOODS_FIELDS
    path = os.path.join(tempfile.gettempdir(), "declaration.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    return path

# ---------------------------
# FastAPI app + download APIs
# ---------------------------
app = FastAPI(title="ECUS App")

@app.get("/download/xml")
def download_xml():
    if not ECUS_GOODS:
        return {"error": "No goods rows"}
    xml_text = fill_template(ECUS_HEADER, ECUS_GOODS)
    path = write_text_file(xml_text, "declaration.xml")
    # Define filename -> Starlette sets Content-Disposition: attachment
    return FileResponse(
        path,
        filename="declaration.xml",
        media_type="application/octet-stream",  # prevent inline pretty-print
        headers={"Content-Disposition": 'attachment; filename="declaration.xml"'}
    )

@app.get("/download/csv")
def download_csv():
    if not ECUS_GOODS:
        return {"error": "No goods rows"}
    path = write_csv_file(ECUS_HEADER, ECUS_GOODS)
    return FileResponse(
        path,
        filename="declaration.csv",
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="declaration.csv"'}
    )

# ---------------------------
# Gradio UI (mounted after routes)
# ---------------------------
def build_ecus_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📦 ECUS Declaration — PDF → Header + Goods → XML/CSV")

        # Upload + preview
        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_preview = gr.HTML()
        pdf_file.upload(show_pdf, inputs=pdf_file, outputs=pdf_preview)

        status = gr.Label()

        with gr.Tabs():
            # ===== Header tab =====
            with gr.Tab("Declaration Header"):
                gr.Markdown("### General")
                with gr.Row():
                    SoToKhai    = gr.Textbox(label="SoToKhai")
                    NgayKhaiBao = gr.Textbox(label="NgayKhaiBao")
                    LoaiHinh    = gr.Textbox(label="LoaiHinh")
                    HaiQuan     = gr.Textbox(label="HaiQuan")

                gr.Markdown("### Exporter")
                with gr.Row():
                    ExporterName        = gr.Textbox(label="ExporterName")
                    ExporterCountryCode = gr.Textbox(label="ExporterCountryCode")
                ExporterAddress = gr.Textbox(label="ExporterAddress")

                gr.Markdown("### Importer")
                with gr.Row():
                    ImporterName    = gr.Textbox(label="ImporterName")
                    ImporterTaxCode = gr.Textbox(label="ImporterTaxCode")
                ImporterAddress = gr.Textbox(label="ImporterAddress")

                gr.Markdown("### Transport")
                with gr.Row():
                    PhuongTien   = gr.Textbox(label="PhuongTien")
                    BillOfLading = gr.Textbox(label="BillOfLading")
                    PortOfEntry  = gr.Textbox(label="PortOfEntry")

                gr.Markdown("### Taxes")
                with gr.Row():
                    ImportDuty = gr.Textbox(label="ImportDuty")
                    VAT        = gr.Textbox(label="VAT")
                    TotalTax   = gr.Textbox(label="TotalTax")

                def save_header(*vals):
                    keys = [
                        "SoToKhai","NgayKhaiBao","LoaiHinh","HaiQuan",
                        "ExporterName","ExporterAddress","ExporterCountryCode",
                        "ImporterName","ImporterAddress","ImporterTaxCode",
                        "PhuongTien","BillOfLading","PortOfEntry",
                        "ImportDuty","VAT","TotalTax"
                    ]
                    for k, v in zip(keys, vals):
                        ECUS_HEADER[k] = v or ""
                    return "✅ Header saved"

                gr.Button("💾 Save Header").click(
                    save_header,
                    inputs=[
                        SoToKhai, NgayKhaiBao, LoaiHinh, HaiQuan,
                        ExporterName, ExporterAddress, ExporterCountryCode,
                        ImporterName, ImporterAddress, ImporterTaxCode,
                        PhuongTien, BillOfLading, PortOfEntry,
                        ImportDuty, VAT, TotalTax
                    ],
                    outputs=[status]
                )

            # ===== Goods tab =====
            with gr.Tab("Goods Items"):
                goods_table = gr.Dataframe(
                    headers=GOODS_FIELDS, datatype="str",
                    row_count=(1, "dynamic"), col_count=(len(GOODS_FIELDS), "dynamic"),
                    interactive=True, label="Goods Table"
                )
                goods_status = gr.Label()

                def add_goods(tbl):
                    if not tbl:
                        return "⚠️ Nothing to add"
                    headers = tbl.headers if hasattr(tbl, "headers") else GOODS_FIELDS
                    rows = tbl.values.tolist() if hasattr(tbl, "values") else tbl
                    added = 0
                    for r in rows:
                        if any(str(x).strip() for x in r):
                            ECUS_GOODS.append(dict(zip(headers, r)))
                            added += 1
                    return f"✅ Added {added} rows (total {len(ECUS_GOODS)})"

                def clear_goods():
                    ECUS_GOODS.clear()
                    return "Cleared goods."

                with gr.Row():
                    gr.Button("➕ Add to Goods").click(add_goods, inputs=[goods_table], outputs=[goods_status])
                    gr.Button("🧹 Clear Goods").click(clear_goods, outputs=[goods_status])

            # ===== Extract & Download tab =====
            with gr.Tab("Extract & Download"):
                def do_extract(file):
                    if file is None:
                        empty_headers = [""] * len(ECUS_HEADER_FIELDS)
                        table_clear = gr.update(value=[], headers=GOODS_FIELDS,
                                                col_count=(len(GOODS_FIELDS), "dynamic"))
                        return ("⚠️ No file", *empty_headers, table_clear)

                    path = file.name if hasattr(file, "name") else file
                    header, goods = extract_from_pdf(path)

                    # Save state
                    ECUS_HEADER.update(header)
                    ECUS_GOODS.clear()
                    ECUS_GOODS.extend(goods)

                    # Map header to outputs (keep order consistent with Save Header inputs)
                    header_vals = [
                        ECUS_HEADER["SoToKhai"], ECUS_HEADER["NgayKhaiBao"],
                        ECUS_HEADER["LoaiHinh"], ECUS_HEADER["HaiQuan"],
                        ECUS_HEADER["ExporterName"], ECUS_HEADER["ExporterAddress"], ECUS_HEADER["ExporterCountryCode"],
                        ECUS_HEADER["ImporterName"], ECUS_HEADER["ImporterAddress"], ECUS_HEADER["ImporterTaxCode"],
                        ECUS_HEADER["PhuongTien"], ECUS_HEADER["BillOfLading"], ECUS_HEADER["PortOfEntry"],
                        ECUS_HEADER["ImportDuty"], ECUS_HEADER["VAT"], ECUS_HEADER["TotalTax"],
                    ]
                    goods_rows = [[g.get(h, "") for h in GOODS_FIELDS] for g in goods]
                    goods_update = gr.update(value=goods_rows, headers=GOODS_FIELDS,
                                             col_count=(len(GOODS_FIELDS), "dynamic"))
                    return ("✅ Extracted", *header_vals, goods_update)

                extract_btn = gr.Button("🧠 Extract from PDF")
                extract_btn.click(
                    do_extract,
                    inputs=[pdf_file],
                    outputs=[
                        status,
                        # 16 header textboxes in this exact order:
                        SoToKhai, NgayKhaiBao, LoaiHinh, HaiQuan,
                        ExporterName, ExporterAddress, ExporterCountryCode,
                        ImporterName, ImporterAddress, ImporterTaxCode,
                        PhuongTien, BillOfLading, PortOfEntry,
                        ImportDuty, VAT, TotalTax,
                        # goods table
                        goods_table
                    ]
                )

                gr.Markdown("### Download files")
                with gr.Row():
                    gr.Button("⬇️ Download XML").click(None, js="() => { window.open('/download/xml', '_blank') }")
                    gr.Button("⬇️ Download CSV").click(None, js="() => { window.open('/download/csv', '_blank') }")

                gr.Markdown(
                    "- Files are generated on-demand from the current in-memory data (header + goods).\n"
                    "- If you haven’t extracted/added any goods, the download endpoints will return a small JSON error."
                )

        return demo

# Mount Gradio AFTER defining the routes (important for downloads)
ui = build_ecus_ui()
app = gr.mount_gradio_app(app, ui, path="/")
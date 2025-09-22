import gradio as gr
import json
import csv
import yaml
import base64
import tempfile
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from utils.helpers import build_prompt, clean_json_output, get_fields_from_table
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Session
session_results, last_fields, table_data = [], None, []

# === Config ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
with open("template/template.xml", "r", encoding="utf-8") as f:
    template_xml = f.read()

DEFAULT_FIELDS = config.get("DEFAULT_FIELDS", ["Field1", "Field2", "Field3"])

# === Core Extraction ===
def extract_info(pdf_file, fields_table):
    global last_fields, table_data
    if pdf_file is None:
        return [], "⚠️ No file uploaded"

    fields_list = get_fields_from_table(fields_table)
    if not fields_list:
        return [], "⚠️ Please add at least one field"

    last_fields = fields_list
    sys_prompt = build_prompt(fields_list)

    path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    uploaded_file = client.files.create(file=open(path, "rb"), purpose="assistants")
    file_id = uploaded_file.id

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user",
                    "content": [
                        {"type": "input_text", "text": sys_prompt},
                        {"type": "input_file", "file_id": file_id}
                    ]}]
        )
        raw_output = resp.output_text.strip()
        output = clean_json_output(raw_output)
        try:
            data = json.loads(output)
        except Exception as e:
            return [], f"⚠️ Failed to parse JSON:\n\n{raw_output}\n\nError: {str(e)}"

        if isinstance(data, dict):
            data = [data]
        if not data:
            return [], "⚠️ No entries extracted."

        rows = [[str(entry.get(h, "")) for h in fields_list] for entry in data]
        table_data = rows
        return gr.update(value=rows, headers=fields_list,
                         col_count=(len(fields_list), "dynamic")), "✅ Extraction successful"
    finally:
        client.files.delete(file_id)

def add_to_results(table):
    global session_results, last_fields
    if table is None or not last_fields:
        return "⚠️ Nothing to add", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))
    if hasattr(table, "values"):
        table = table.values.tolist()
    entries = [dict(zip(last_fields, row)) for row in table if any(row)]
    session_results.extend(entries)

    if session_results:
        headers = list(session_results[0].keys())
        rows = [[str(e.get(h, "")) for h in headers] for e in session_results]
        return (
            f"✅ Added {len(entries)} entries (total {len(session_results)})",
            gr.update(value=rows, headers=headers, col_count=(len(headers),"dynamic"))
        )
    return "⚠️ Nothing to add", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))

# === File writers ===
def write_results_files():
    if not session_results:
        return None, None
    json_path = os.path.join(tempfile.gettempdir(), "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(tempfile.gettempdir(), "results.csv")
    headers = session_results[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(session_results)
    return json_path, csv_path

def write_xml_results(template_xml: str):
    if not session_results:
        return None

    # Build goods items block
    goods_items = []
    for entry in session_results:
        goods_items.append(f"""
        <Item>
            <Description>{entry.get("Description","")}</Description>
            <HSCode>{entry.get("HSCode","")}</HSCode>
            <OriginCountry>{entry.get("OriginCountry","")}</OriginCountry>
            <Quantity>{entry.get("Quantity","")}</Quantity>
            <Unit>{entry.get("Unit","")}</Unit>
            <CIFValue>{entry.get("CIFValue","")}</CIFValue>
            <Currency>{entry.get("Currency","")}</Currency>
        </Item>
        """)
    goods_block = "\n".join(goods_items)

    # Use the first row for "header-level" fields (GeneralInfo, Exporter, Importer, Transport, Taxes)
    first = session_results[0]

    xml_filled = template_xml.format(
        SoToKhai=first.get("SoToKhai", ""),
        NgayKhaiBao=first.get("NgayKhaiBao", ""),
        LoaiHinh=first.get("LoaiHinh", ""),
        HaiQuan=first.get("HaiQuan", ""),

        ExporterName=first.get("ExporterName", ""),
        ExporterAddress=first.get("ExporterAddress", ""),
        ExporterCountryCode=first.get("ExporterCountryCode", ""),

        ImporterName=first.get("ImporterName", ""),
        ImporterAddress=first.get("ImporterAddress", ""),
        ImporterTaxCode=first.get("ImporterTaxCode", ""),

        PhuongTien=first.get("PhuongTien", ""),
        BillOfLading=first.get("BillOfLading", ""),
        PortOfEntry=first.get("PortOfEntry", ""),

        GoodsList=goods_block,

        ImportDuty=first.get("ImportDuty", ""),
        VAT=first.get("VAT", ""),
        TotalTax=first.get("TotalTax", "")
    )

    xml_path = os.path.join(tempfile.gettempdir(), "results.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_filled)

    return xml_path

def clear_results():
    global session_results, table_data
    session_results, table_data = [], []
    return [], "Cleared all results.", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))

# === PDF Preview ===
def show_pdf(file):
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

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧾 Multi-entry PDF Information Extraction")

    with gr.Row():
        with gr.Column(scale=2):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_preview = gr.HTML(label="Preview PDF")
        with gr.Column(scale=1):
            gr.Markdown("### Fields (edit, add, remove below)")
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

    gr.Examples(
        examples=[
            ["data/DN2305.015_sample.pdf"],
            ["data/new_format_1.pdf"],
            ["data/new_format_3.pdf"],
            ["data/new_format_5.pdf"]
        ],
        fn=show_pdf,
        inputs=[pdf_file],
        outputs=[pdf_preview],
        cache_examples=False,
        label="Try with example PDFs",
        preload=True
    )

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

    add_results_btn = gr.Button("Add to Results")

    session_table = gr.Dataframe(
        headers=["Field1"],
        datatype="str",
        row_count=(1, "dynamic"),
        col_count=(1, "dynamic"),
        interactive=False,
        label="Accumulated Results"
    )

    with gr.Row():
        download_json_btn = gr.Button("⬇️ Download JSON")
        download_csv_btn = gr.Button("⬇️ Download CSV")
        download_xml_btn = gr.Button("⬇️ Download XML")
        clear_btn = gr.Button("Clear Session")

        download_json_btn.click(None, js="() => { window.open('/download/json', '_blank') }")
        download_csv_btn.click(None, js="() => { window.open('/download/csv', '_blank') }")
        download_xml_btn.click(None, js="() => { window.open('/download/xml', '_blank') }")

    extract_btn.click(extract_info, inputs=[pdf_file, fields_table],
                      outputs=[demo_table, status])
    add_results_btn.click(add_to_results, inputs=[demo_table],
                          outputs=[status, session_table])
    clear_btn.click(clear_results,
                    outputs=[demo_table, status, session_table])

# === FastAPI integration ===
app = FastAPI()

@app.get("/download/json")
def download_json_api():
    json_path, _ = write_results_files()
    if not json_path:
        return {"error": "No results to download"}
    return FileResponse(json_path, filename="results.json", media_type="application/json")

@app.get("/download/csv")
def download_csv_api():
    _, csv_path = write_results_files()
    if not csv_path:
        return {"error": "No results to download"}
    return FileResponse(csv_path, filename="results.csv", media_type="text/csv")

@app.get("/download/xml")
def download_xml_api():
    xml_path = write_xml_results(template_xml)
    if not xml_path:
        return {"error": "No results to download"}
    return FileResponse(xml_path, filename="results.xml", media_type="application/xml")

# Mount Gradio after endpoints
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("demo:app", host="0.0.0.0", port=7860, reload=True)
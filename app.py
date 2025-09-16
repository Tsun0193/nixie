import gradio as gr
import json, re, csv, base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Session
session_results, last_fields, table_data = [], None, []

DEFAULT_FIELDS = [
    "Số phiếu", "Ngày chứng từ(mm/dd/yyyy)", "Ngày xuất(mm/dd/yyyy)",
    "Mã FG", "Đvt", "Số lượng(Thùng)", "Số lượng(Pcs)",
    "Kho nhận", "Địa chỉ", "Mã AR"
]

# === Helper ===
def build_prompt(custom_fields):
    fields_text = "\n".join([f'    \"{f}\": ,' for f in custom_fields])
    return f"""
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
Chỉ trả về JSON hợp lệ, không thêm giải thích.
"""

def clean_json_output(output: str) -> str:
    output = re.sub(r"^```[a-zA-Z]*\n?", "", output.strip())
    output = re.sub(r"\n?```$", "", output.strip())
    return output.strip()

def get_fields_from_table(table):
    if table is None:
        return []
    if hasattr(table, "values"):  # pandas DataFrame
        table = table.values.tolist()
    return [row[0].strip() for row in table if row and row[0] and str(row[0]).strip()]

# === Core ===
def extract_info(pdf_file, fields_table):
    global last_fields, table_data
    if pdf_file is None:
        return [], "⚠️ No file uploaded"

    fields_list = get_fields_from_table(fields_table)
    if not fields_list:
        return [], "⚠️ Please add at least one field"

    last_fields = fields_list
    sys_prompt = build_prompt(fields_list)

    uploaded_file = client.files.create(file=open(pdf_file.name, "rb"), purpose="assistants")
    file_id = uploaded_file.id

    try:
        resp = client.responses.create(
            model="gpt-4o",
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
            return [], "⚠️ No entries extracted"

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

def download_json():
    if not session_results: return None
    path = "extracted_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_results, f, ensure_ascii=False, indent=2)
    return path

def download_csv():
    if not session_results: return None
    path = "extracted_results.csv"
    headers = session_results[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader(); writer.writerows(session_results)
    return path

def clear_results():
    global session_results, table_data
    session_results, table_data = [], []
    return [], "Cleared all results.", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))

# === PDF Preview (base64 embed, A4 ratio) ===
def show_pdf(file):
    if file is None:
        return ""
    with open(file.name, "rb") as f:
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

# === UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧾 Multi-entry PDF Information Extraction")

    # Row 1: Upload (trái) + Fields (phải)
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

    # Bảng accumulated (dummy header để init)
    session_table = gr.Dataframe(
        headers=["Field1"],
        datatype="str",
        row_count=(1, "dynamic"),
        col_count=(1, "dynamic"),
        interactive=False,
        label="Accumulated Results"
    )

    with gr.Row():
        download_json_btn = gr.Button("Download JSON")
        download_csv_btn = gr.Button("Download CSV")
        clear_btn = gr.Button("Clear Session")

    download_json_file = gr.File(label="Download JSON", interactive=False)
    download_csv_file = gr.File(label="Download CSV", interactive=False)

    # Wiring
    extract_btn.click(extract_info, inputs=[pdf_file, fields_table],
                      outputs=[demo_table, status])
    add_results_btn.click(add_to_results, inputs=[demo_table],
                          outputs=[status, session_table])
    download_json_btn.click(download_json, outputs=[download_json_file])
    download_csv_btn.click(download_csv, outputs=[download_csv_file])
    clear_btn.click(clear_results,
                    outputs=[demo_table, status, session_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
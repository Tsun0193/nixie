from fastapi import FastAPI
from fastapi.responses import FileResponse
import gradio as gr
import os, json, csv, tempfile

# ======= Minimal in-memory store =======
MAIN_ROWS = []

DEFAULT_FIELDS = [
    "Số phiếu", "Ngày chứng từ(mm/dd/yyyy)", "Ngày xuất(mm/dd/yyyy)",
    "Mã FG", "Đvt", "Số lượng(Thùng)", "Số lượng(Pcs)",
    "Kho nhận", "Địa chỉ", "Mã AR"
]

def write_json(rows):
    path = os.path.join(tempfile.gettempdir(), "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path

def write_csv(rows):
    if not rows:
        raise ValueError("No rows")
    path = os.path.join(tempfile.gettempdir(), "results.csv")
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader(); w.writerows(rows)
    return path

# ======= Gradio UI (Main) =======
def build_main_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 Multi-entry PDF Information Extraction (Main)")

        fields_df = gr.Dataframe(
            headers=["Fields"], datatype="str",
            row_count=(len(DEFAULT_FIELDS), "dynamic"),
            col_count=(1, "fixed"),
            value=[[f] for f in DEFAULT_FIELDS],
            interactive=True, label="Fields (edit)"
        )

        table = gr.Dataframe(
            headers=DEFAULT_FIELDS, datatype="str",
            row_count=(1, "dynamic"),
            col_count=(len(DEFAULT_FIELDS), "dynamic"),
            interactive=True, label="Extracted Entries (editable)"
        )

        status = gr.Label()

        def add_rows(tbl):
            if not tbl:
                return "⚠️ Nothing to add"
            headers = tbl.headers if hasattr(tbl, "headers") else DEFAULT_FIELDS
            values = tbl.values.tolist() if hasattr(tbl, "values") else tbl
            for row in values:
                if any(str(x).strip() for x in row):
                    MAIN_ROWS.append(dict(zip(headers, row)))
            return f"✅ Total rows: {len(MAIN_ROWS)}"

        gr.Button("Add to Results").click(add_rows, inputs=[table], outputs=[status])

        with gr.Row():
            gr.Button("⬇️ Download JSON").click(None, js="() => { window.open('/download/json','_blank') }")
            gr.Button("⬇️ Download CSV").click(None, js="() => { window.open('/download/csv','_blank') }")
            def clear():
                MAIN_ROWS.clear()
                return "Cleared."
            gr.Button("Clear").click(clear, outputs=[status])

    return demo

# ======= FastAPI app (Main) =======
app = FastAPI(title="Main IE App")
main_ui = build_main_ui()
app = gr.mount_gradio_app(app, main_ui, path="/")

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
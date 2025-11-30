import gradio as gr
from typing import Tuple, Any
from app.settings import DEFAULT_FIELDS
from app.services.extraction import extract_from_pdf
from app.state.memory import main_store
from app.ui.common import show_pdf

def build_main_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 Multi-entry PDF Information Extraction")

        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                pdf_preview = gr.HTML(label="Preview PDF")
            with gr.Column(scale=1):
                gr.Markdown("### Fields (edit below)")
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

        session_table = gr.Dataframe(
            headers=["Field1"],
            datatype="str",
            row_count=(1, "dynamic"),
            col_count=(1, "dynamic"),
            interactive=False,
            label="Accumulated Results"
        )

        extract_btn.click(
            _extract_handler,
            inputs=[pdf_file, fields_table],
            outputs=[demo_table, status]
        )

        add_results_btn.click(
            _add_results_handler,
            inputs=[demo_table],
            outputs=[status, session_table]
        )

        return demo

def _extract_handler(pdf_file, fields_table):
    if pdf_file is None:
        return [], "⚠️ No file uploaded"
    # table -> headers list
    fields_list = [row[0] for row in (fields_table or []) if row and row[0]]
    if not fields_list:
        return [], "⚠️ Please add at least one field"

    path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    try:
        merged_rows, _, _ = extract_from_pdf(path, fields_list)
    except Exception as e:
        return [], f"⚠️ Extraction failed: {e}"

    if not merged_rows:
        return [], "⚠️ No entries extracted."

    rows = [[str(entry.get(h, "")) for h in fields_list] for entry in merged_rows]
    return gr.update(value=rows, headers=fields_list,
                     col_count=(len(fields_list), "dynamic")), "✅ Extraction successful"

def _add_results_handler(table):
    if table is None or not table:
        return "⚠️ Nothing to add", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))

    headers = table.headers if hasattr(table, "headers") else None
    values = table.values.tolist() if hasattr(table, "values") else table

    if not headers or not values:
        return "⚠️ Nothing to add", gr.update(value=[], headers=["Field1"], col_count=(1,"dynamic"))

    main_store.add_rows(headers, values)
    rows = main_store.get_rows()
    out_headers = list(rows[0].keys()) if rows else ["Field1"]
    out_values = [[str(r.get(h, "")) for h in out_headers] for r in rows]
    return f"✅ Added {len(values)} entries (total {len(rows)})", \
           gr.update(value=out_values, headers=out_headers, col_count=(len(out_headers), "dynamic"))

# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import gradio as gr

from app.ui.main import build_main_ui
from app.ui.ecus import build_ecus_ui
from app.state.memory import main_store, ecus_store
from app.services.files import write_json, write_csv, write_text
from app.services.xml_builder import fill_template
from app.settings import TEMPLATE_XML

app = FastAPI(title="Nixie IE", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["GET","POST"], allow_headers=["*"],
)

# --- downloads (main) ---
@app.get("/download/json")
def download_json_api():
    rows = main_store.get_rows()
    if not rows: return {"error":"No results"}
    return FileResponse(write_json(rows, "results.json"),
                        filename="results.json", media_type="application/json")

@app.get("/download/csv")
def download_csv_api():
    rows = main_store.get_rows()
    if not rows: return {"error":"No results"}
    return FileResponse(write_csv(rows, "results.csv"),
                        filename="results.csv", media_type="text/csv")

# --- downloads (ecus) ---
@app.get("/ecus/download/xml")
def download_ecus_xml_api():
    rows = ecus_store.get_rows()
    if not rows: return {"error":"No declaration rows"}
    xml_text = fill_template(TEMPLATE_XML, rows)
    return FileResponse(write_text(xml_text, "declaration.xml"),
                        filename="declaration.xml", media_type="application/xml")

@app.get("/ecus/download/csv")
def download_ecus_csv_api():
    rows = ecus_store.get_rows()
    if not rows: return {"error":"No declaration rows"}
    return FileResponse(write_csv(rows, "declaration.csv"),
                        filename="declaration.csv", media_type="text/csv")

# --- mount gradio UIs ---
main_ui = build_main_ui()   # Blocks()
ecus_ui = build_ecus_ui()   # Blocks()

# IMPORTANT: reuse the SAME `app`
app = gr.mount_gradio_app(app, main_ui, path="/")        # main UI at /
app = gr.mount_gradio_app(app, ecus_ui, path="/ecus")    # ECUS UI at /ecus
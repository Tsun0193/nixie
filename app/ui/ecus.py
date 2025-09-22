import gradio as gr
from typing import List, Dict
from app.services.ecus_extraction import (
    extract_ecus_from_pdf, ECUS_HEADER_FIELDS, ECUS_GOODS_FIELDS
)
from app.state.memory import ecus_store
from app.ui.common import show_pdf
from app.services.xml_builder import fill_template
from app.services.files import write_csv, write_text
from app.settings import TEMPLATE_XML

ECUS_HEADER: Dict[str, str] = {k: "" for k in ECUS_HEADER_FIELDS}
ECUS_GOODS: List[Dict[str, str]] = []

def build_ecus_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📦 ECUS Declaration — PDF → Header + Goods → XML/CSV")

        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_preview = gr.HTML()
        pdf_file.upload(show_pdf, inputs=pdf_file, outputs=pdf_preview)

        extract_status = gr.Label()

        # ===== Header controls (organized) =====
        # Keep the order in sync with ECUS_HEADER_FIELDS groups
        # General
        SoToKhai      = gr.Textbox(label="SoToKhai")
        NgayKhaiBao   = gr.Textbox(label="NgayKhaiBao")
        LoaiHinh      = gr.Textbox(label="LoaiHinh")
        HaiQuan       = gr.Textbox(label="HaiQuan")
        # Exporter
        ExporterName        = gr.Textbox(label="ExporterName")
        ExporterAddress     = gr.Textbox(label="ExporterAddress")
        ExporterCountryCode = gr.Textbox(label="ExporterCountryCode")
        # Importer
        ImporterName    = gr.Textbox(label="ImporterName")
        ImporterAddress = gr.Textbox(label="ImporterAddress")
        ImporterTaxCode = gr.Textbox(label="ImporterTaxCode")
        # Transport
        PhuongTien   = gr.Textbox(label="PhuongTien")
        BillOfLading = gr.Textbox(label="BillOfLading")
        PortOfEntry  = gr.Textbox(label="PortOfEntry")
        # Taxes
        ImportDuty = gr.Textbox(label="ImportDuty")
        VAT        = gr.Textbox(label="VAT")
        TotalTax   = gr.Textbox(label="TotalTax")

        # Lay out nicely with Tabs
        with gr.Tabs():
            with gr.Tab("Declaration Header"):
                gr.Markdown("### General Info")
                with gr.Row():
                    SoToKhai; NgayKhaiBao; LoaiHinh; HaiQuan
                gr.Markdown("### Exporter")
                with gr.Row():
                    ExporterName; ExporterCountryCode
                ExporterAddress.render()
                gr.Markdown("### Importer")
                with gr.Row():
                    ImporterName; ImporterTaxCode
                ImporterAddress.render()
                gr.Markdown("### Transport")
                with gr.Row():
                    PhuongTien; BillOfLading; PortOfEntry
                gr.Markdown("### Taxes")
                with gr.Row():
                    ImportDuty; VAT; TotalTax

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
                    outputs=[extract_status]
                )

            with gr.Tab("Goods Items"):
                goods_table = gr.Dataframe(
                    headers=ECUS_GOODS_FIELDS,
                    datatype="str",
                    row_count=(1, "dynamic"),
                    col_count=(len(ECUS_GOODS_FIELDS), "dynamic"),
                    interactive=True,
                    label="Goods Table"
                )
                goods_status = gr.Label()

                def add_goods(tbl):
                    if not tbl:
                        return "⚠️ Nothing to add"
                    headers = tbl.headers if hasattr(tbl, "headers") else ECUS_GOODS_FIELDS
                    values = tbl.values.tolist() if hasattr(tbl, "values") else tbl
                    added = 0
                    for row in values:
                        if any(str(x).strip() for x in row):
                            ECUS_GOODS.append(dict(zip(headers, row)))
                            added += 1
                    return f"✅ Added {added} goods rows (total {len(ECUS_GOODS)})"

                def clear_goods():
                    ECUS_GOODS.clear()
                    return "Cleared goods."

                with gr.Row():
                    gr.Button("➕ Add to Goods").click(add_goods, inputs=[goods_table], outputs=[goods_status])
                    gr.Button("🧹 Clear Goods").click(clear_goods, outputs=[goods_status])

            with gr.Tab("Extract & Export"):
                def do_extract(file):
                    if file is None:
                        return ("⚠️ No file",) + tuple([gr.update()] * (len(ECUS_HEADER_FIELDS) + 1))
                    path = file.name if hasattr(file, "name") else file
                    header, goods = extract_ecus_from_pdf(path)

                    # Save global state
                    ECUS_HEADER.clear(); ECUS_HEADER.update(header)
                    ECUS_GOODS.clear(); ECUS_GOODS.extend(goods)

                    # Map header dict to controls (in same order as inputs above)
                    header_vals = [ECUS_HEADER[k] for k in [
                        "SoToKhai","NgayKhaiBao","LoaiHinh","HaiQuan",
                        "ExporterName","ExporterAddress","ExporterCountryCode",
                        "ImporterName","ImporterAddress","ImporterTaxCode",
                        "PhuongTien","BillOfLading","PortOfEntry",
                        "ImportDuty","VAT","TotalTax"
                    ]]

                    goods_rows = [[g.get(h, "") for h in ECUS_GOODS_FIELDS] for g in goods]
                    goods_update = gr.update(value=goods_rows, headers=ECUS_GOODS_FIELDS,
                                             col_count=(len(ECUS_GOODS_FIELDS),"dynamic"))

                    return ("✅ Extracted", *header_vals, goods_update)

                extract_btn = gr.Button("🧠 Extract from PDF")
                extract_btn.click(
                    do_extract,
                    inputs=[pdf_file],
                    outputs=[
                        extract_status,
                        SoToKhai, NgayKhaiBao, LoaiHinh, HaiQuan,
                        ExporterName, ExporterAddress, ExporterCountryCode,
                        ImporterName, ImporterAddress, ImporterTaxCode,
                        PhuongTien, BillOfLading, PortOfEntry,
                        ImportDuty, VAT, TotalTax,
                        goods_table
                    ]
                )

                export_status = gr.Label()

                def build_xml():
                    if not ECUS_GOODS:
                        return "⚠️ No goods rows"
                    xml_str = fill_template(TEMPLATE_XML, [{"GoodsList": ""},])  # unused; we pass header separately below
                    # we actually need xml from header+goods; use xml_builder’s fill_template directly:
                    xml_text = fill_template(TEMPLATE_XML, rows=[{**ECUS_HEADER, **g} for g in ECUS_GOODS] or [ECUS_HEADER])
                    write_text(xml_text, "declaration.xml")
                    return "✅ XML ready. Open /ecus/download/xml"

                def build_csv():
                    if not ECUS_GOODS:
                        return "⚠️ No goods rows"
                    # repeat header on each row for CSV
                    rows = [{**ECUS_HEADER, **g} for g in ECUS_GOODS]
                    write_csv(rows, fname="declaration.csv")
                    return "✅ CSV ready. Open /ecus/download/csv"

                with gr.Row():
                    gr.Button("📄 Build XML").click(build_xml, outputs=[export_status])
                    gr.Button("🧾 Build CSV").click(build_csv, outputs=[export_status])

                gr.Markdown("- XML: `/ecus/download/xml`\n- CSV: `/ecus/download/csv`")

        return demo
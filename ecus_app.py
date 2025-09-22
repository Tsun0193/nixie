from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.ui.ecus import build_ecus_ui
from app.services.files import write_csv, write_text
import gradio as gr
import os, csv, tempfile

# ======= In-memory state =======
ECUS_HEADER = {  # single declaration header
    "SoToKhai": "", "NgayKhaiBao": "", "LoaiHinh": "", "HaiQuan": "",
    "ExporterName": "", "ExporterAddress": "", "ExporterCountryCode": "",
    "ImporterName": "", "ImporterAddress": "", "ImporterTaxCode": "",
    "PhuongTien": "", "BillOfLading": "", "PortOfEntry": "",
    "ImportDuty": "", "VAT": "", "TotalTax": "",
}
ECUS_GOODS = []  # list[dict] of item rows

GOODS_FIELDS = ["Description", "HSCode", "OriginCountry", "Quantity", "Unit", "CIFValue", "Currency"]

# ======= Template =======
TEMPLATE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Declaration>
    <GeneralInfo>
        <SoToKhai>{SoToKhai}</SoToKhai>
        <NgayKhaiBao>{NgayKhaiBao}</NgayKhaiBao>
        <LoaiHinh>{LoaiHinh}</LoaiHinh>
        <HaiQuan>{HaiQuan}</HaiQuan>
    </GeneralInfo>
    <Exporter>
        <Name>{ExporterName}</Name>
        <Address>{ExporterAddress}</Address>
        <CountryCode>{ExporterCountryCode}</CountryCode>
    </Exporter>
    <Importer>
        <Name>{ImporterName}</Name>
        <Address>{ImporterAddress}</Address>
        <TaxCode>{ImporterTaxCode}</TaxCode>
    </Importer>
    <Transport>
        <PhuongTien>{PhuongTien}</PhuongTien>
        <BillOfLading>{BillOfLading}</BillOfLading>
        <PortOfEntry>{PortOfEntry}</PortOfEntry>
    </Transport>
    <GoodsList>
{GoodsList}
    </GoodsList>
    <Taxes>
        <ImportDuty>{ImportDuty}</ImportDuty>
        <VAT>{VAT}</VAT>
        <TotalTax>{TotalTax}</TotalTax>
    </Taxes>
</Declaration>
"""

def build_goods_items(rows):
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
    return "\n".join(chunks) if chunks else ""

def fill_template(header, goods_rows):
    mapping = {**header, "GoodsList": build_goods_items(goods_rows)}
    return TEMPLATE_XML.format(**mapping)

def write_file(text: str, name: str):
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def write_csv(header: dict, goods_rows: list[dict]):
    if not goods_rows:
        raise ValueError("No goods rows")
    out_rows = []
    for g in goods_rows:
        row = {**header, **g}  # header repeated per item
        out_rows.append(row)
    # stable header order: header keys first, then goods fields
    headers = list(header.keys()) + GOODS_FIELDS
    path = os.path.join(tempfile.gettempdir(), "declaration.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader(); w.writerows(out_rows)
    return path

# ======= Gradio UI (Header form + Goods table) =======
def build_ecus_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📦 ECUS Declaration — One Declaration, Multiple Goods")

        with gr.Tabs():
            with gr.Tab("Declaration Header"):
                gr.Markdown("### General Info")
                with gr.Row():
                    SoToKhai = gr.Textbox(label="Số tờ khai (SoToKhai)", value=ECUS_HEADER["SoToKhai"])
                    NgayKhaiBao = gr.Textbox(label="Ngày khai báo (NgayKhaiBao)", value=ECUS_HEADER["NgayKhaiBao"])
                    LoaiHinh = gr.Textbox(label="Loại hình (LoaiHinh)", value=ECUS_HEADER["LoaiHinh"])
                    HaiQuan = gr.Textbox(label="Hải quan (HaiQuan)", value=ECUS_HEADER["HaiQuan"])

                gr.Markdown("### Exporter")
                with gr.Row():
                    ExporterName = gr.Textbox(label="Name", value=ECUS_HEADER["ExporterName"])
                    ExporterCountryCode = gr.Textbox(label="CountryCode", value=ECUS_HEADER["ExporterCountryCode"])
                ExporterAddress = gr.Textbox(label="Address", value=ECUS_HEADER["ExporterAddress"])

                gr.Markdown("### Importer")
                with gr.Row():
                    ImporterName = gr.Textbox(label="Name", value=ECUS_HEADER["ImporterName"])
                    ImporterTaxCode = gr.Textbox(label="TaxCode", value=ECUS_HEADER["ImporterTaxCode"])
                ImporterAddress = gr.Textbox(label="Address", value=ECUS_HEADER["ImporterAddress"])

                gr.Markdown("### Transport")
                with gr.Row():
                    PhuongTien = gr.Textbox(label="Phương tiện (PhuongTien)", value=ECUS_HEADER["PhuongTien"])
                    BillOfLading = gr.Textbox(label="BillOfLading", value=ECUS_HEADER["BillOfLading"])
                    PortOfEntry = gr.Textbox(label="PortOfEntry", value=ECUS_HEADER["PortOfEntry"])

                gr.Markdown("### Taxes")
                with gr.Row():
                    ImportDuty = gr.Textbox(label="ImportDuty", value=ECUS_HEADER["ImportDuty"])
                    VAT = gr.Textbox(label="VAT", value=ECUS_HEADER["VAT"])
                    TotalTax = gr.Textbox(label="TotalTax", value=ECUS_HEADER["TotalTax"])

                status_header = gr.Label()

                def save_header(*vals):
                    keys = list(ECUS_HEADER.keys())
                    new_vals = {
                        "SoToKhai": vals[0], "NgayKhaiBao": vals[1], "LoaiHinh": vals[2], "HaiQuan": vals[3],
                        "ExporterName": vals[4], "ExporterAddress": vals[6], "ExporterCountryCode": vals[5],
                        "ImporterName": vals[7], "ImporterAddress": vals[9], "ImporterTaxCode": vals[8],
                        "PhuongTien": vals[10], "BillOfLading": vals[11], "PortOfEntry": vals[12],
                        "ImportDuty": vals[13], "VAT": vals[14], "TotalTax": vals[15],
                    }
                    ECUS_HEADER.update(new_vals)
                    return "✅ Header saved"

                gr.Button("💾 Save Header").click(
                    save_header,
                    inputs=[
                        SoToKhai, NgayKhaiBao, LoaiHinh, HaiQuan,
                        ExporterName, ExporterCountryCode, ExporterAddress,
                        ImporterName, ImporterTaxCode, ImporterAddress,
                        PhuongTien, BillOfLading, PortOfEntry,
                        ImportDuty, VAT, TotalTax
                    ],
                    outputs=[status_header]
                )

            with gr.Tab("Goods Items"):
                gr.Markdown("### Goods (add/edit rows below)")
                goods_table = gr.Dataframe(
                    headers=GOODS_FIELDS,
                    datatype="str",
                    row_count=(1, "dynamic"),
                    col_count=(len(GOODS_FIELDS), "dynamic"),
                    interactive=True,
                    label="Goods Table (Description, HSCode, OriginCountry, Quantity, Unit, CIFValue, Currency)"
                )
                status_goods = gr.Label()

                def add_goods(tbl):
                    if not tbl:
                        return "⚠️ Nothing to add"
                    headers = tbl.headers if hasattr(tbl, "headers") else GOODS_FIELDS
                    values = tbl.values.tolist() if hasattr(tbl, "values") else tbl
                    added = 0
                    for row in values:
                        if any(str(x).strip() for x in row):
                            ECUS_GOODS.append(dict(zip(headers, row))); added += 1
                    return f"✅ Added {added} goods rows (total {len(ECUS_GOODS)})"

                def clear_goods():
                    ECUS_GOODS.clear()
                    return "Cleared goods."

                with gr.Row():
                    gr.Button("➕ Add to Goods").click(add_goods, inputs=[goods_table], outputs=[status_goods])
                    gr.Button("🧹 Clear Goods").click(clear_goods, outputs=[status_goods])

            with gr.Tab("Export"):
                gr.Markdown("### Export Files")
                export_status = gr.Label()

                def _export_xml():
                    if not ECUS_GOODS:
                        return "⚠️ No goods rows"
                    xml_text = fill_template(ECUS_HEADER, ECUS_GOODS)
                    path = write_file(xml_text, "declaration.xml")
                    # open in new tab via frontend button (below)
                    return "✅ XML ready"

                def _export_csv():
                    if not ECUS_GOODS:
                        return "⚠️ No goods rows"
                    write_csv(ECUS_HEADER, ECUS_GOODS)
                    return "✅ CSV ready"

                with gr.Row():
                    gr.Button("📄 Build XML").click(_export_xml, outputs=[export_status])
                    gr.Button("🧾 Build CSV").click(_export_csv, outputs=[export_status])

                gr.Markdown(
                    "After building files:\n\n"
                    "- **Download XML:** open `/download/xml`\n"
                    "- **Download CSV:** open `/download/csv`"
                )

    return demo

# ======= FastAPI app (ECUS) =======
app = FastAPI(title="ECUS App (Header + Goods)")
ecus_ui = build_ecus_ui()
app = gr.mount_gradio_app(app, build_ecus_ui(), path="/ecus")

@app.get("/ecus/download/xml")
def download_ecus_xml_api():
    rows = [{**ECUS_HEADER, **g} for g in ECUS_GOODS] or [ECUS_HEADER]
    xml_text = fill_template(TEMPLATE_XML, rows)
    path = write_text(xml_text, "declaration.xml")
    return FileResponse(path, filename="declaration.xml", media_type="application/xml")

@app.get("/ecus/download/csv")
def download_ecus_csv_api():
    rows = [{**ECUS_HEADER, **g} for g in ECUS_GOODS]
    if not rows:
        return {"error": "No declaration rows"}
    path = write_csv(rows, fname="declaration.csv")
    return FileResponse(path, filename="declaration.csv", media_type="text/csv")
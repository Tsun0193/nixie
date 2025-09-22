from typing import List, Dict, Any

ECUS_HEADER_FIELDS = [
    "SoToKhai", "NgayKhaiBao", "LoaiHinh", "HaiQuan",
    "ExporterName", "ExporterAddress", "ExporterCountryCode",
    "ImporterName", "ImporterAddress", "ImporterTaxCode",
    "PhuongTien", "BillOfLading", "PortOfEntry",
    "ImportDuty", "VAT", "TotalTax",
]

ECUS_GOODS_FIELDS = [
    "Description", "HSCode", "OriginCountry", "Quantity", "Unit", "CIFValue", "Currency"
]

def build_goods_items(rows: List[Dict[str, Any]]) -> str:
    chunks = []
    for r in rows:
        chunks.append(
f"""        <Item>
            <Description>{r.get('Description','')}</Description>
            <HSCode>{r.get('HSCode','')}</HSCode>
            <OriginCountry>{r.get('OriginCountry','')}</OriginCountry>
            <Quantity>{r.get('Quantity','')}</Quantity>
            <Unit>{r.get('Unit','')}</Unit>
            <CIFValue>{r.get('CIFValue','')}</CIFValue>
            <Currency>{r.get('Currency','')}</Currency>
        </Item>"""
        )
    return "\n".join(chunks)

def fill_template(template_xml: str, rows: List[Dict[str, Any]]) -> str:
    """
    One Declaration; many goods. Header values are taken from the first row.
    """
    if not rows:
        raise ValueError("No rows provided for ECUS XML")

    hdr = rows[0]
    goods_block = build_goods_items(rows)

    # .format mapping
    mapping = {
        "SoToKhai": hdr.get("SoToKhai",""),
        "NgayKhaiBao": hdr.get("NgayKhaiBao",""),
        "LoaiHinh": hdr.get("LoaiHinh",""),
        "HaiQuan": hdr.get("HaiQuan",""),

        "ExporterName": hdr.get("ExporterName",""),
        "ExporterAddress": hdr.get("ExporterAddress",""),
        "ExporterCountryCode": hdr.get("ExporterCountryCode",""),

        "ImporterName": hdr.get("ImporterName",""),
        "ImporterAddress": hdr.get("ImporterAddress",""),
        "ImporterTaxCode": hdr.get("ImporterTaxCode",""),

        "PhuongTien": hdr.get("PhuongTien",""),
        "BillOfLading": hdr.get("BillOfLading",""),
        "PortOfEntry": hdr.get("PortOfEntry",""),

        "GoodsList": goods_block,

        "ImportDuty": hdr.get("ImportDuty",""),
        "VAT": hdr.get("VAT",""),
        "TotalTax": hdr.get("TotalTax",""),
    }

    return template_xml.format(**mapping)
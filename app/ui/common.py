import base64

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
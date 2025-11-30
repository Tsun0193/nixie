import json
from app.clients.openai_client import get_openai_client
from app.settings import settings
from utils.helpers import build_prompt, clean_json_output, parse_model_payload 

def extract_from_pdf(file_path: str, fields: list[str]) -> tuple[list[dict], dict | None, list[dict]]:
    """
    Calls the LLM with a prompt + file.
    Returns (merged_rows, metadata_row_or_none, table_rows).
    """
    client = get_openai_client()
    prompt = build_prompt(fields)

    uploaded = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    file_id = uploaded.id
    try:
        resp = client.responses.create(
            model=settings.model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id}
                ]
            }]
        )
        raw_output = resp.output_text.strip()
        output = clean_json_output(raw_output)

        data = json.loads(output)
        merged_rows, metadata_row, table_rows = parse_model_payload(data, fields)
        return merged_rows, metadata_row, table_rows
    finally:
        client.files.delete(file_id)

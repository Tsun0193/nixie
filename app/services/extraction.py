import json
from app.clients.openai_client import get_openai_client
from app.settings import settings
from utils.helpers import build_prompt, clean_json_output 

def extract_from_pdf(file_path: str, fields: list[str]) -> list[dict]:
    """
    Calls the LLM with a prompt + file. Returns list of dicts (rows).
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
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("Model response is not a list/dict")

        return data
    finally:
        client.files.delete(file_id)
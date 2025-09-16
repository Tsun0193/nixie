from paddleocr import PaddleOCR, PPStructureV3
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import cv2
import yaml
import json
import re
from bs4 import BeautifulSoup
import numpy as np
from matplotlib import pyplot as plt

MODEL_ID = "google/gemma-3-270m"
device = "cuda:2" if torch.cuda.is_available() else "cpu"
ocr = PPStructureV3()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

FIELDS = config["fields"]
PROMPT_SYSTEM = config["PROMPT_SYSTEM"]

pages = convert_from_path("data/DN2305.015_sample.pdf", dpi=400)
img = np.array(pages[0])
plt.imshow(img)
plt.axis('off')
plt.show()

result = ocr.predict(
    input=img,
    use_doc_orientation_classify=False,
    use_table_recognition=True,
    use_doc_unwarping=False
)


def extract(result): 
    all_pages = [] 
    for page in result: # iterate over pages 
        page_texts = [] 
        for item in page['parsing_res_list']: 
            c = item.content 
            if isinstance(c, str) and c.strip().startswith("<html"): 
                soup = BeautifulSoup(c, "html.parser") 
                for row in soup.find_all("tr"): 
                    row_text = " ".join(td.get_text(" ", strip=True) for td in row.find_all("td")) 
                    if row_text.strip(): 
                        page_texts.append(row_text) 
            else: 
                raw = str(c).strip() # drop page markers like "1/1", "2/5", etc. 
                if not re.fullmatch(r"\d+/\d+", raw): 
                    page_texts.append(raw) 
        all_pages.append(page_texts) 
    return all_pages

result = extract(result)
out = ""
for p in result:
    out += "\n".join(p) + "\n\n"
print(out)


def correct(text: str):
    """
    Use the model to correct OCR text.
    Args:
        text (str): raw OCR text
    Returns:
        str: corrected text
    """
    # build prompt
    prompt = PROMPT_SYSTEM.format(text=text)

    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,              # deterministic
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # cut off the prompt tokens → keep only generated part
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][input_len:]
    corrected_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return corrected_text

final = correct(out)
print(final)
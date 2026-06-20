import requests, base64
from dotenv import load_dotenv
import os
import pymupdf

load_dotenv()

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

doc = pymupdf.open("book.pdf")
doc.select([11, 15]) 

image_content_blocks = []
for page in doc:
    pix = page.get_pixmap(dpi=150)        
    img_bytes = pix.tobytes("png")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    image_content_blocks.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64}"
        }
    })

headers = {
    "Authorization": f"Bearer {os.getenv('QWEN_API_KEY')}",
    "Accept": "application/json"
}

payload = {
    "model": "qwen/qwen3.5-397b-a17b",
    "messages": [
        {
            "role": "system",
            "content": "You are an expert book reader. You will be given contents of a book and your job is to extract the chapter names and their page ranges."
        },
        {
            "role": "user",
            "content": [
                *image_content_blocks,
                {
                    "type": "text",
                    "text": "Extract the chapter names and their page ranges from these pages."
                }
            ]
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "chapters",
            "schema": {
                "type": "object",
                "properties": {
                    "chapters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chapter_name": {"type": "string"},
                                "start_page":   {"type": "integer"},
                                "end_page":     {"type": "integer"}
                            },
                            "required": ["chapter_name", "start_page", "end_page"]
                        }
                    }
                },
                "required": ["chapters"]
            }
        }
    },
    "max_tokens": 16384,
    "temperature": 0.60,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "stream": False,
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=False)
print(response.json())
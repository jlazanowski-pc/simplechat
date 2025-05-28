"""
functions_vision.py
Lightweight wrapper around your GPT-4o-mini vision deployment.
"""

from __future__ import annotations
from typing import Union
from openai import AzureOpenAI
from PIL import Image
import base64, io, os, functools

def _b64(data: Union[bytes, Image.Image]) -> str:
    if isinstance(data, Image.Image):
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        data = buf.getvalue()
    return base64.b64encode(data).decode()

@functools.lru_cache
def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_VISION_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_VISION_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_VISION_VERSION", "2024-02-15-preview"),
    )

def analyze_image(
    
    image: Union[bytes, Image.Image],
    prompt: str = "Describe the image in detail for later retrieval.",
    deployment: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    
    print("We're in analyze_image")
    model_name = deployment or os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o-mini-vision")
    print(f"[Vision] using deployment: {model_name}")
    res = _client().chat.completions.create(
        model=deployment or os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",  "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}},
            ],
        }],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content.strip()

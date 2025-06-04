"""
functions_vision.py
Lightweight wrapper around your GPT-4o-mini vision deployment.
"""

from __future__ import annotations
from typing import Union
from openai import AzureOpenAI
from PIL import Image
import base64, io, functools

from config import *
from functions_settings import *

def _b64(data: Union[bytes, Image.Image]) -> str:
    if isinstance(data, Image.Image):
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        data = buf.getvalue()
    return base64.b64encode(data).decode()

@functools.lru_cache
def _client() -> AzureOpenAI:
    settings = get_settings()
    if settings.get("enable_vision_apim"):
        return AzureOpenAI(
            api_version=settings.get("azure_apim_vision_api_version"),
            azure_endpoint=settings.get("azure_apim_vision_endpoint"),
            api_key=settings.get("azure_apim_vision_subscription_key"),
        )

    if settings.get("azure_openai_vision_authentication_type") == "managed_identity":
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            api_version=settings.get("azure_openai_vision_api_version"),
            azure_endpoint=settings.get("azure_openai_vision_endpoint"),
            azure_ad_token_provider=token_provider,
        )

    return AzureOpenAI(
        api_version=settings.get("azure_openai_vision_api_version"),
        azure_endpoint=settings.get("azure_openai_vision_endpoint"),
        api_key=settings.get("azure_openai_vision_key"),
    )

def analyze_image(
    
    image: Union[bytes, Image.Image],
    prompt: str = "Describe the image in detail for later retrieval.",
    deployment: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    
    settings = get_settings()
    model_name = deployment

    if not model_name:
        if settings.get("enable_vision_apim"):
            model_name = settings.get("azure_apim_vision_deployment")
        else:
            vision_model_obj = settings.get("vision_model", {})
            if vision_model_obj and vision_model_obj.get("selected"):
                selected = vision_model_obj["selected"][0]
                model_name = selected.get("deploymentName")
        model_name = model_name or settings.get("azure_openai_vision_deployment")

    print(f"[Vision] using deployment: {model_name}")
    res = _client().chat.completions.create(
        model=model_name,
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

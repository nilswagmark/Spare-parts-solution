import time
from typing import Any, Dict, Optional

import base64
import json

import httpx

from ..config import Settings


PROMPT = """
You are an expert grill part inspector. Decide if rust on the part is
cleanable surface rust or deep corrosion that requires replacement.

Context from field photos (stainless flavorizer bars):
- Deep corrosion/replace: perforations or pinholes, missing metal, heavy
  pitting, rough/scaly surfaces, flaking sheets, large dark cavities, or edges
  eaten away. Several provided examples show through-holes and thick, layered
  scale on the tops and sides of the bars.
- Cleanable surface rust: uniform discoloration or thin film, light spotting,
  no visible metal loss, smooth surface shape preserved (may show orange/brown
  tint but geometry is intact).
- Stainless steel can show heat tint or minor surface oxidation; prefer
  cleanable if the surface stays smooth and intact.

Return JSON with keys:
- classification: "cleanable_surface_rust" | "deep_corrosion_replace" | "uncertain"
- confidence: 0-1
- rationale: short, focused on pitting/holes/flaking vs light film

Rules:
- If there is pitting, holes, flaking, or material loss => deep_corrosion_replace.
- If discoloration/film only and no material loss => cleanable_surface_rust.
- If the view is unclear, cropped too tight, or the part is not visible enough
  to judge integrity => uncertain with low confidence.
Output JSON only. No markdown.
"""


class GeminiClient:
    """
    Despite the name, this client now talks to OpenAI GPT vision models.
    The rest of the app can stay unchanged; configuration is driven by
    OPENAI_API_KEY and OPENAI_MODEL in the environment.
    """

    def __init__(self, settings: Settings):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.settings = settings

    async def classify_image(self, image_bytes: bytes, part_type: Optional[str]) -> Dict[str, Any]:
        """
        Call the OpenAI chat completions API with image input.

        Demo/offline mode has been removed so that every request goes through
        the VLM/LLM. If no API key is configured, this will raise rather than
        silently returning a placeholder.
        """
        start = time.time()

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured; cannot call OpenAI.")

        # Encode image as base64 for data URL.
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_b64}"

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_text = PROMPT + (f"\n\nPart type: {part_type}" if part_type else "")

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            # Ask OpenAI to respond with a JSON object so we can parse it easily.
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.time() - start) * 1000)

        # OpenAI returns the JSON as a string in message.content.
        try:
            choices = data.get("choices", [])
            first = choices[0] if choices else {}
            message = first.get("message", {})
            text = message.get("content", "")
            parsed = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parse
            raise RuntimeError(f"Failed to parse OpenAI response: {data}") from exc

        # Attach latency and model version for downstream logic.
        parsed["latency_ms"] = latency_ms
        parsed.setdefault("model_version", self.model)
        return parsed

    async def classify_image_url(self, image_url: str, part_type: Optional[str]) -> Dict[str, Any]:
        """
        Same as classify_image, but takes a publicly accessible image URL.
        This is useful for platforms (like Mavenoid) that can host the image
        and pass only a URL to this backend.
        """
        start = time.time()

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured; cannot call OpenAI.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt_text = PROMPT + (f"\n\nPart type: {part_type}" if part_type else "")

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.time() - start) * 1000)

        try:
            choices = data.get("choices", [])
            first = choices[0] if choices else {}
            message = first.get("message", {})
            text = message.get("content", "")
            parsed = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parse
            raise RuntimeError(f"Failed to parse OpenAI response: {data}") from exc

        parsed["latency_ms"] = latency_ms
        parsed.setdefault("model_version", self.model)
        return parsed


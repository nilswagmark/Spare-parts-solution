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
    def __init__(self, settings: Settings):
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.settings = settings

    async def classify_image(self, image_bytes: bytes, part_type: Optional[str]) -> Dict[str, Any]:
        """
        Call the real Gemini vision endpoint.

        Demo/offline mode has been removed so that every request goes through
        the VLM/LLM. If no API key is configured, this will raise rather than
        silently returning a placeholder.
        """
        start = time.time()

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured; cannot call Gemini.")

        # Encode image as base64 for Gemini inline_data.
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct Gemini generateContent request.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        body: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT + (f"\n\nPart type: {part_type}" if part_type else "")},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.time() - start) * 1000)

        # Gemini returns text; our prompt instructs it to output JSON.
        try:
            candidates = data.get("candidates", [])
            first = candidates[0] if candidates else {}
            parts = first.get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            parsed = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parse
            raise RuntimeError(f"Failed to parse Gemini response: {data}") from exc

        # Attach latency and model version for downstream logic.
        parsed["latency_ms"] = latency_ms
        parsed.setdefault("model_version", self.model)
        return parsed


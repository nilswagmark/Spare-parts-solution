import time
from typing import Any, Dict, Optional

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
        start = time.time()

        if self.settings.demo_mode or not self.api_key:
            # Offline/demo path: deterministic placeholder for local testing
            latency_ms = int((time.time() - start) * 1000)
            return {
                "classification": "cleanable_surface_rust",
                "confidence": 0.5,
                "rationale": "Demo mode: no API key present.",
                "model_version": f"{self.model}-demo",
                "latency_ms": latency_ms,
            }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg"),
        }
        data = {
            "prompt": PROMPT,
            "part_type": part_type or "",
            "model": self.model,
        }

        # NOTE: replace the URL with the official Gemini Vision endpoint when wiring this up.
        url = "https://generativelanguage.googleapis.com/v1beta/vision:classify-rust"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()

        latency_ms = int((time.time() - start) * 1000)
        payload["latency_ms"] = latency_ms
        return payload


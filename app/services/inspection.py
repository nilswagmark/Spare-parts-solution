from typing import Optional

from ..config import Settings
from ..models import InspectionResult
from ..clients.gemini import GeminiClient
from ..utils.image import preprocess_image_to_bytes


async def _build_result(raw: dict, settings: Settings) -> InspectionResult:
    classification = raw.get("classification", "uncertain")
    confidence = float(raw.get("confidence", 0.0))
    rationale = raw.get("rationale", "No rationale returned.")
    latency_ms = int(raw.get("latency_ms", 0))
    model_version = raw.get("model_version", settings.openai_model)

    needs_review = (
        classification == "uncertain" or confidence < settings.confidence_threshold
    )

    return InspectionResult(
        classification=classification,
        confidence=confidence,
        rationale=rationale,
        needs_review=needs_review,
        model_version=model_version,
        latency_ms=latency_ms,
    )


async def inspect_image(
    image_bytes: bytes,
    part_type: Optional[str],
    settings: Settings,
    client: GeminiClient,
) -> InspectionResult:
    processed = preprocess_image_to_bytes(image_bytes, settings.max_image_size_px)
    raw = await client.classify_image(processed, part_type)
    return await _build_result(raw, settings)


async def inspect_image_url(
    image_url: str,
    part_type: Optional[str],
    settings: Settings,
    client: GeminiClient,
) -> InspectionResult:
    raw = await client.classify_image_url(image_url, part_type)
    return await _build_result(raw, settings)


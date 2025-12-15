from typing import Optional

from pydantic import BaseModel, Field


class InspectionRequest(BaseModel):
    part_type: Optional[str] = Field(
        default=None, description="Optional hint: burner_tube, grate, flavorizer_bar, etc.",
    )


class InspectionResult(BaseModel):
    classification: str = Field(
        description="cleanable_surface_rust | deep_corrosion_replace | uncertain",
        examples=["cleanable_surface_rust"],
    )
    confidence: float = Field(ge=0.0, le=1.0, example=0.83)
    rationale: str = Field(description="Short explanation of why the decision was made.")
    needs_review: bool = Field(
        description="True when confidence below threshold or model was uncertain."
    )
    model_version: str = Field(description="Underlying VLM model identifier.")
    latency_ms: int = Field(description="End-to-end model call latency in milliseconds.")


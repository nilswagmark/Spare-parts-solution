from functools import lru_cache
from typing import Optional

from pydantic import BaseModel
import os


class Settings(BaseModel):
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
    max_image_size_px: int = int(os.getenv("MAX_IMAGE_SIZE_PX", "1024"))
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
    api_token: Optional[str] = os.getenv("API_TOKEN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


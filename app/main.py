from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .models import InspectionRequest, InspectionResult
from .clients.gemini import GeminiClient
from .services.inspection import inspect_image

app = FastAPI(title="Rust Inspector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_token(
    settings: Settings = Depends(get_settings),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """
    Simple bearer token auth. Set API_TOKEN in the environment and call with:
    Authorization: Bearer <token>
    """
    if not settings.api_token:
        # If no token configured, allow all (useful for local dev).
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(status_code=403, detail="Invalid API token")


def get_client(settings: Settings = Depends(get_settings)) -> GeminiClient:
    return GeminiClient(settings)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectionResult, dependencies=[Depends(require_api_token)])
async def inspect(
    request: InspectionRequest = Depends(),
    file: UploadFile = File(..., description="Image file of the part"),
    settings: Settings = Depends(get_settings),
    client: GeminiClient = Depends(get_client),
) -> InspectionResult:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        result = await inspect_image(image_bytes, request.part_type, settings, client)
    except Exception as exc:  # pragma: no cover - top-level protection
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return result


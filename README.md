# Rust Inspector FastAPI service

Lightweight backend that wraps a vision model (Gemini) to classify grill part rust.

## Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment (at minimum):
```bash
export GEMINI_API_KEY=your_key
export GEMINI_MODEL=gemini-1.5-pro
export CONFIDENCE_THRESHOLD=0.72
export MAX_IMAGE_SIZE_PX=1024
# optional local stub without a key
export DEMO_MODE=true
```

## Run
```bash
uvicorn app.main:app --reload --port 8001
```

## API
- `GET /healthz`
- `POST /inspect` (multipart form): `file` (image), optional `part_type`.
  - Response: `classification`, `confidence`, `rationale`, `needs_review`, `model_version`, `latency_ms`.

## Wire up Gemini
Replace the placeholder endpoint in `app/clients/gemini.py` with the official Gemini Vision endpoint and payload shape. The service will fall back to a demo response if `DEMO_MODE=true` or no API key is supplied.


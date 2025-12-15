import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from app.clients.gemini import GeminiClient
from app.config import get_settings
from app.services.inspection import inspect_image


ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "eval" / "cases.json"
RESULTS_PATH = ROOT / "eval" / "results.json"


async def run_case(
    case: Dict[str, Any], settings, client: GeminiClient
) -> Dict[str, Any]:
    image_path = ROOT / case["image"]
    if not image_path.exists():
        return {
            "id": case.get("id"),
            "status": "missing_file",
            "image": str(image_path),
            "expected": case.get("expected"),
            "notes": case.get("notes", ""),
        }

    with image_path.open("rb") as f:
        image_bytes = f.read()

    try:
        result = await inspect_image(
            image_bytes=image_bytes,
            part_type=case.get("part_type"),
            settings=settings,
            client=client,
        )
        payload = result.model_dump()
        payload.update(
            {
                "id": case.get("id"),
                "image": str(image_path),
                "expected": case.get("expected"),
                "notes": case.get("notes", ""),
                "match": (
                    payload["classification"] == case.get("expected")
                    if case.get("expected")
                    else None
                ),
                "status": "ok",
            }
        )
        return payload
    except Exception as exc:  # pragma: no cover - eval helper
        return {
            "id": case.get("id"),
            "status": "error",
            "error": str(exc),
            "image": str(image_path),
            "expected": case.get("expected"),
            "notes": case.get("notes", ""),
        }


async def main() -> None:
    if not CASES_PATH.exists():
        raise SystemExit(f"Missing cases file: {CASES_PATH}")

    cases: List[Dict[str, Any]] = json.loads(CASES_PATH.read_text())
    settings = get_settings()
    client = GeminiClient(settings)

    results = []
    for case in cases:
        results.append(await run_case(case, settings, client))

    matches = sum(1 for r in results if r.get("match") is True)
    mismatches = sum(1 for r in results if r.get("match") is False)
    missing = sum(1 for r in results if r.get("status") == "missing_file")
    errors = sum(1 for r in results if r.get("status") == "error")

    summary = {
        "total_cases": len(results),
        "matches": matches,
        "mismatches": mismatches,
        "missing_files": missing,
        "errors": errors,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            {"timestamp": int(time.time()), "summary": summary, "results": results},
            indent=2,
        )
    )

    print(
        f"Eval complete: {summary['matches']} match, "
        f"{summary['mismatches']} mismatch, "
        f"{summary['missing_files']} missing, "
        f"{summary['errors']} errors "
        f"({len(results)} cases)."
    )
    print(f"Details saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())


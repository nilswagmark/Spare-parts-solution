Evaluation harness (manual images)
=================================

This folder holds a small manifest of corrosion examples for stainless
flavorizer bars. To run the eval, save the five provided reference photos into
the indicated paths, then execute the script.

Steps
-----
- Save the attached photos to the paths listed in `cases.json` (create
  subfolders as needed).
- From the repo root, run: `python scripts/run_eval.py`
- Results are written to `eval/results.json` and printed to stdout.

Notes
-----
- Files that are missing will be reported as `missing_file` and skipped.
- `expected` labels are best-effort; if you disagree, adjust `cases.json`.
- The script calls `inspect_image`, so the same preprocessing and client
  settings are used as the API. Set `GEMINI_API_KEY` and `GEMINI_MODEL`, or
  use `DEMO_MODE=true` for offline deterministic output.


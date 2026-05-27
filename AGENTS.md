# AGENTS.md

This file is the default working guide for coding agents in this repository.
The project is a TV/video ad detector with a human review loop. Preserve the
reviewable workflow and be careful with boundary semantics.

## Project Shape

- `scripts/detect_ads.py` is a thin compatibility CLI entrypoint. Keep existing
  command-line behavior stable because Docker, NAS deployments, and web flows
  call it directly.
- Core detection code lives under `scripts/ad_detector/`.
- `scripts/ad_detector/pipeline.py` orchestrates one video:
  sample -> template match -> auto discovery -> boundary refine -> review assets.
- `scripts/ad_detector/boundary/` owns boundary alignment and validation.
  `BoundaryRefiner` is the entrypoint.
- `scripts/ad_detector/review_assets.py` writes CSV, screenshots, JSON/debug
  outputs, and review frame paths.
- `scripts/review_ads.py` builds/updates `output/ad_review.xlsx`.
- `scripts/cut_ads.py` reads the review workbook and writes cleaned videos.
- `scripts/web_app.py` is the simple NAS-friendly web UI.
- `scripts/package_docker_bundle.py` creates the zip used for NAS deployment.

## Boundary Semantics

Do not casually change these meanings:

- `start`: first ad frame.
- `end`: last ad frame shown in review screenshots.
- `end_after` / `end_after_seconds`: first non-ad/content frame after the ad,
  and the right-side cut boundary.
- Cutting should remove `[start, end_after)`.
- Review screenshots should mean:
  - `start_before`: frame before the ad start, normally content.
  - `start`: first ad frame.
  - `middle`: inside the ad.
  - `end`: last ad frame.
  - `end_after`: first content frame, or the placeholder `__VIDEO_END__`.

Avoid reintroducing scattered `+1 frame` / `-1 frame` adjustments. If a boundary
needs both sides, carry both fields explicitly.

## Boundary Refinement Rules

- `diff` is only a candidate cut signal. It means "the picture changed", not
  "this is an ad boundary".
- `visual` confirms whether the ad side of a candidate actually looks like an
  ad segment.
- Start refinement should prefer candidates whose right-side window has strong
  ad visual score.
- End refinement should be symmetric: the left side should look like ad, and the
  right side should stop looking like ad/content should resume.
- Template matching is for recall only. Do not assume the template source start
  is the exact boundary for a new episode; per-episode inserts may differ.
- Padding is not a primary fix for bad boundaries because it can delete real
  content. Prefer real boundary detection and reviewable debug output.

## Debug Fields

Boundary decisions are intentionally surfaced in CSV/Excel through
`boundary_debug`. Keep it compact JSON rather than adding many columns.
Useful keys include:

- `start_score`, `end_score`
- `start_visual_score`, `end_visual_score`
- `start_expanded_attempt`, `start_expanded_used`
- `end_expanded_attempt`, `end_expanded_used`
- `start_seconds_before`, `end_seconds_before`

When adding new diagnostics, prefer one JSON field under `boundary_debug`.

## Review Workbook And Web UI

- `output/ad_review.xlsx` is the source of truth for cutting.
- The web UI edits the same workbook. Keep offline Excel and web review behavior
  compatible.
- Input and output file lists support selection. Avoid breaking multi-select
  flows.
- Review rows can be generated incrementally. Do not overwrite user edits when
  appending later rows.
- Avoid global page polling that resets table scroll or collapsed file groups.
  Logs may refresh independently; review content should refresh on explicit
  events or long-poll style update endpoints.
- Image URLs should work through NAS reverse proxies. Do not rely on query
  parameters such as `?inline=1` for preview images.

## Docker / NAS Constraints

- `docker-compose.yml` is intended to run as a stable service in Synology
  Container Manager.
- The bundle zip should contain all files needed to build/run locally on NAS:
  `docker-compose.yml`, `Dockerfile`, `requirements.txt`, `scripts/`,
  `ad_templates/`, `input/`, `output/`.
- `scripts/` is mounted so bug fixes can be deployed without rebuilding the
  image when possible.
- Do not depend on `git` being available inside Docker build contexts on NAS.

## Performance Notes

- Sampling is often the largest cost. The default sampling mode currently uses
  `--sample-skip-frame noref` to reduce decode work while preserving enough
  signal for detection.
- Boundary refinement uses higher-rate local frame reads. Keep candidate counts
  bounded and avoid launching ffmpeg for every possible cut when sampled metrics
  are enough for rough scoring.
- Hardware decode is not assumed by default. Do not add hardware-specific ffmpeg
  flags unless there is a clear fallback.

## Verification Commands

After any code change, verify before reporting completion. Do not treat an edit
as finished until relevant checks have run, or explicitly explain why a check
could not be run.

Tests and regression checks should run in the Docker environment by default,
because NAS deployment uses the container runtime. Local checks are useful for
fast feedback, but Docker verification is the baseline for completion.

Run a quick local static compilation first when useful:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tv_ad_detector_pycache python3 -m py_compile \
  scripts/detect_ads.py scripts/build_review.py scripts/review_ads.py \
  scripts/cut_ads.py scripts/web_app.py scripts/ad_detector/*.py \
  scripts/ad_detector/boundary/*.py
```

Then run static compilation in Docker:

```bash
docker compose run --rm web python -m py_compile \
  scripts/detect_ads.py scripts/build_review.py scripts/review_ads.py \
  scripts/cut_ads.py scripts/web_app.py scripts/ad_detector/*.py \
  scripts/ad_detector/boundary/*.py
```

Useful regression videos/cases used during development:

- Episode 7 around `5:58-6:18`: start should be ad, `end` should be last ad
  frame, `end_after` should be first content frame.
- Episode 9 around `28:12`: `28:12` is a strong content cut but not an ad start;
  visual confirmation should push start to about `28:16`.
- Episodes 1, 3, 13 around the repeated mid-roll ads have historically caught
  start/before-start and end/after-end regressions.

Example focused run:

```bash
docker compose run --rm -e PYTHONPATH=/work/scripts web \
  python scripts/detect_ads.py --input-dir input --output-dir output/detect_check \
  --template-dir ad_templates --ignore-keypoints \
  --files '《山花烂漫时2024》电视剧第09集.mp4'
```

## Git Hygiene

- The worktree may contain user-generated outputs and local test files. Do not
  delete or revert unrelated files.
- `output/`, `dist/`, generated zips, and local videos are artifacts unless the
  user explicitly asks otherwise.
- Before committing, inspect `git status --short` and stage only intentional
  source/config/doc changes.

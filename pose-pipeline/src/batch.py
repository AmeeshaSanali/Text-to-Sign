#!/usr/bin/env python3
"""
Batch Pipeline Runner (Phases 2-4)
Part of Track B - Phase 6: Scale Up

Runs end-to-end extraction, cleaning, and retargeting over manifest rows.
Supports batching, failure logging, and status updating in data/manifest.csv.
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import Optional, List

# Add pose-pipeline/src to import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import extract_video_landmarks
from clean import clean_landmark_data
from retarget import retarget_and_export_json


def run_batch_pipeline(
    dataset_root: Path = Path("Dataset"),
    manifest_path: Path = Path("data/manifest.csv"),
    raw_dir: Path = Path("data/landmarks_raw"),
    clean_dir: Path = Path("data/landmarks_clean"),
    anim_dir: Path = Path("data/animations/slsl"),
    overlay_dir: Optional[Path] = None,
    target_status: str = "todo",
    limit: Optional[int] = None,
    specific_glosses: Optional[List[str]] = None,
    mark_as: str = "retargeted",
):
    print("=== SLSL Batch Pipeline Runner (Extract -> Clean -> Retarget) ===")
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Read manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    # Filter rows to process
    queue = []
    for r in rows:
        gloss = r["gloss"]
        status = r.get("status", "todo").strip().lower()

        if specific_glosses:
            if gloss in specific_glosses:
                queue.append(r)
        elif target_status == "all" or status == target_status.lower():
            queue.append(r)

    if limit is not None and limit > 0:
        queue = queue[:limit]

    print(f"Found {len(queue)} signs to process in this batch.\n")
    if not queue:
        print("Nothing to process.")
        return

    successes = []
    failures = []

    for idx, row in enumerate(queue, 1):
        gloss = row["gloss"]
        take = row["take"]
        src_video = dataset_root / row["source_video"]
        mirrored = row.get("mirrored", "False").lower() == "true"

        print(f"[{idx}/{len(queue)}] Processing {gloss} ({take})...")

        if not src_video.exists():
            print(f"  [FAIL] Source video not found: {src_video}")
            failures.append({"gloss": gloss, "take": take, "reason": "video_not_found"})
            continue

        try:
            # 1. Extraction (Phase 2)
            out_raw = raw_dir / gloss / f"{take}.npz"
            out_overlay = (overlay_dir / f"{gloss}_{take}_overlay.mp4") if overlay_dir else None
            extract_stats = extract_video_landmarks(
                video_path=src_video,
                out_path=out_raw,
                overlay_path=out_overlay,
            )

            # 2. Cleaning (Phase 3)
            out_clean = clean_dir / gloss / f"{take}.npz"
            clean_stats = clean_landmark_data(
                in_npz_path=out_raw,
                out_npz_path=out_clean,
                target_fps=30.0,
                trim_idle=True,
                mirrored=mirrored,
            )

            # 3. Retargeting (Phase 4)
            out_json = anim_dir / f"{gloss}.json"
            retarget_stats = retarget_and_export_json(
                clean_npz_path=out_clean,
                out_json_path=out_json,
                gloss_name=gloss,
            )

            # Update status in row
            row["status"] = mark_as
            successes.append({
                "gloss": gloss,
                "take": take,
                "frames": retarget_stats["frames"],
                "duration": retarget_stats["duration"],
                "lh_lost": extract_stats["lh_lost_pct"],
                "rh_lost": extract_stats["rh_lost_pct"],
            })
            print(f"  [OK] Exported {out_json.name} ({retarget_stats['frames']} frames, {retarget_stats['duration']}s)")

        except Exception as e:
            print(f"  [FAIL] Error processing {gloss}: {e}", file=sys.stderr)
            failures.append({"gloss": gloss, "take": take, "reason": str(e)})

    # Write back updated manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Batch Pipeline Execution Summary ===")
    print(f"Total Attempted: {len(queue)}")
    print(f"Successful:      {len(successes)}")
    print(f"Failed:          {len(failures)}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f['gloss']} ({f['take']}): {f['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Run batch extraction, cleaning, and retargeting.")
    parser.add_argument("--dataset", default="Dataset", help="Dataset root path (default: 'Dataset')")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Manifest path")
    parser.add_argument("--status", default="todo", help="Filter signs by manifest status (default: 'todo', or 'all')")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of signs to process")
    parser.add_argument("--glosses", default=None, help="Comma-separated list of specific glosses to process")
    parser.add_argument("--mark-as", default="retargeted", help="Status to assign in manifest upon success (default: 'retargeted')")
    parser.add_argument("--render-overlays", action="store_true", help="Also render debug overlay videos")
    args = parser.parse_args()

    specific = [g.strip().upper() for g in args.glosses.split(",")] if args.glosses else None
    overlay_dir = Path("data/videos/batch_overlays") if args.render_overlays else None

    run_batch_pipeline(
        dataset_root=Path(args.dataset).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        target_status=args.status,
        limit=args.limit,
        specific_glosses=specific,
        mark_as=args.mark_as,
        overlay_dir=overlay_dir,
    )


if __name__ == "__main__":
    main()

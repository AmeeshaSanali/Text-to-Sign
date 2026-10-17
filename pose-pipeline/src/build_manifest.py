#!/usr/bin/env python3
"""
Manifest Builder for SSL400 Dataset
Part of Track B - Phase 1: Vocabulary and Manifest

Generates data/manifest.csv with normalized gloss names, metadata, and canonical take selections.
"""

import os
import sys
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict


# Pre-vetted starter set of 10 words with audited optimal takes
STARTER_SET = {
    "HELLO": {
        "category": "Greetings",
        "label": "Hello",
        "take": "Hello_049",
        "notes": "starter_set; high visibility take (0.98 wrist vis)",
    },
    "THANK_YOU": {
        "category": "Greetings",
        "label": "Thank you",
        "take": "Thank you_058",
        "notes": "starter_set; high visibility take (0.99 wrist vis)",
    },
    "YES": {
        "category": "Interjection",
        "label": "Yes",
        "take": "Yes_002",
        "notes": "starter_set; high visibility take (0.92 wrist vis)",
    },
    "NO": {
        "category": "Determiner",
        "label": "No",
        "take": "No_003",
        "notes": "starter_set; highest visibility take (0.58 wrist vis)",
    },
    "GO": {
        "category": "Verbs",
        "label": "Go",
        "take": "Go_034",
        "notes": "starter_set; high visibility take (0.97 wrist vis)",
    },
    "COME": {
        "category": "Verbs",
        "label": "Come",
        "take": "Come_030",
        "notes": "starter_set; high visibility take (0.99 wrist vis)",
    },
    "EAT": {
        "category": "Verbs",
        "label": "Eat",
        "take": "Eat_021",
        "notes": "starter_set; high visibility take (0.93 wrist vis)",
    },
    "MOTHER": {
        "category": "People",
        "label": "Mother",
        "take": "Mother_027",
        "notes": "starter_set; high visibility take (1.00 wrist vis)",
    },
    "TODAY": {
        "category": "Days",
        "label": "Today",
        "take": "Today_036",
        "notes": "starter_set; high visibility take (1.00 wrist vis)",
    },
    "BLUE": {
        "category": "Colors",
        "label": "Blue",
        "take": "Blue_001",
        "notes": "starter_set; high visibility take (1.00 wrist vis)",
    },
}


def normalize_gloss(label: str) -> str:
    """Standardizes dataset folder names into clean gloss identifiers."""
    name = label.strip()
    match = re.match(r"^\d+\.\s*(.+)$", name)
    if match:
        name = match.group(1).strip()
    if name == "Not like (dislike)":
        return "DISLIKE"
    if name == "Cant":
        return "CANT"
    if name == "Dont":
        return "DONT"
    if name == "Dont know":
        return "DONT_KNOW"
    cleaned = re.sub(r"[^\w\s]", "", name)
    cleaned = re.sub(r"\s+", "_", cleaned).upper()
    return cleaned


def find_best_take(orig_dir: Path, csv_dir: Path = None) -> tuple[Path, str, str]:
    """Finds best take file for a sign based on MediaPipe wrist visibility if available."""
    video_files = sorted([f for f in orig_dir.iterdir() if f.suffix.lower() in [".mp4", ".mov"]])
    if not video_files:
        return None, "", "no_videos_found"

    if not csv_dir or not csv_dir.exists():
        return video_files[0], video_files[0].stem, "first_available"

    best_file = video_files[0]
    best_stem = video_files[0].stem
    best_vis = -1.0

    for vf in video_files:
        cf = csv_dir / f"{vf.stem}.csv"
        if not cf.exists():
            continue
        try:
            with open(cf, "r") as fp:
                rows = list(csv.reader(fp))
            if not rows:
                continue
            lw_list = [float(r[15].rstrip("]").split(",")[-1]) for r in rows if len(r) >= 17]
            rw_list = [float(r[16].rstrip("]").split(",")[-1]) for r in rows if len(r) >= 17]
            if lw_list and rw_list:
                vis = (sum(lw_list) / len(lw_list) + sum(rw_list) / len(rw_list)) / 2.0
                if vis > best_vis:
                    best_vis = vis
                    best_file = vf
                    best_stem = vf.stem
        except Exception:
            continue

    notes = f"selected_by_visibility ({best_vis:.2f})" if best_vis >= 0 else "default_selection"
    return best_file, best_stem, notes


def build_manifest(dataset_root: Path, output_file: Path, include_all_takes: bool = False):
    print(f"=== Building Manifest from: {dataset_root} ===")
    orig_path = dataset_root / "Dataset - Original"
    csv_path = dataset_root / "Dataset - MP - CSV"

    if not orig_path.exists():
        print(f"Error: Original videos path not found at: {orig_path}", file=sys.stderr)
        sys.exit(1)

    categories = sorted([d.name for d in orig_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    rows = []

    header = [
        "gloss",
        "label_in_dataset",
        "category",
        "language",
        "source_video",
        "take",
        "signer_id",
        "mirrored",
        "license",
        "status",
        "notes"
    ]

    starter_glosses = set(STARTER_SET.keys())

    for cat in categories:
        cat_dir = orig_path / cat
        cat_csv_dir = csv_path / cat if csv_path.exists() else None
        sign_dirs = sorted([d.name for d in cat_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

        for sign in sign_dirs:
            sign_dir = cat_dir / sign
            sign_csv_dir = cat_csv_dir / sign if cat_csv_dir else None
            gloss = normalize_gloss(sign)

            if include_all_takes:
                # Add all video takes for this sign
                video_files = sorted([f for f in sign_dir.iterdir() if f.suffix.lower() in [".mp4", ".mov"]])
                for vf in video_files:
                    take = vf.stem
                    is_starter_best = (gloss in starter_glosses and STARTER_SET[gloss]["take"] == take)
                    notes = STARTER_SET[gloss]["notes"] if is_starter_best else ("starter_candidate" if gloss in starter_glosses else "")
                    rel_source = str(vf.relative_to(dataset_root))
                    rows.append({
                        "gloss": gloss,
                        "label_in_dataset": sign,
                        "category": cat,
                        "language": "slsl",
                        "source_video": rel_source,
                        "take": take,
                        "signer_id": "unknown",
                        "mirrored": "False",
                        "license": "unconfirmed",
                        "status": "todo",
                        "notes": notes
                    })
            else:
                # Add canonical single best take per sign
                if gloss in starter_glosses and STARTER_SET[gloss]["category"] == cat:
                    target_take = STARTER_SET[gloss]["take"]
                    matched_vf = None
                    for vf in sign_dir.iterdir():
                        if vf.stem == target_take and vf.suffix.lower() in [".mp4", ".mov"]:
                            matched_vf = vf
                            break
                    if not matched_vf:
                        matched_vf, target_take, _ = find_best_take(sign_dir, sign_csv_dir)
                    take = target_take
                    notes = STARTER_SET[gloss]["notes"]
                    v_file = matched_vf
                else:
                    v_file, take, notes = find_best_take(sign_dir, sign_csv_dir)

                rel_source = str(v_file.relative_to(dataset_root)) if v_file else ""
                rows.append({
                    "gloss": gloss,
                    "label_in_dataset": sign,
                    "category": cat,
                    "language": "slsl",
                    "source_video": rel_source,
                    "take": take,
                    "signer_id": "unknown",
                    "mirrored": "False",
                    "license": "unconfirmed",
                    "status": "todo",
                    "notes": notes
                })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written with {len(rows)} entries to: {output_file}")
    if not include_all_takes:
        starter_count = sum(1 for r in rows if r["gloss"] in starter_glosses)
        print(f"Configured {starter_count} starter signs in manifest.")


def main():
    parser = argparse.ArgumentParser(description="Build dataset manifest for SSL400 sign classes.")
    parser.add_argument("dataset_root", nargs="?", default="Dataset", help="Root directory of Dataset (default: 'Dataset')")
    parser.add_argument("--output", default="data/manifest.csv", help="Output path for manifest CSV")
    parser.add_argument("--all-takes", action="store_true", help="Include all individual takes instead of canonical best take per sign")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()
    build_manifest(dataset_path, output_path, include_all_takes=args.all_takes)


if __name__ == "__main__":
    main()

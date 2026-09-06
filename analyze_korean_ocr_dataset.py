#!/usr/bin/env python3
"""Analyze per-writer label duplication in the Korean word OCR dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = (("Training", "2.단어"), ("Validation", "2.단어"))


def writer_sort_key(path: Path):
    """Sort numeric writer directories numerically, then fall back to names."""
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def analyze_writer(split: str, writer_dir: Path) -> dict:
    image_count = sum(
        1 for path in writer_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    json_paths = sorted(path for path in writer_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    labels = []
    errors = []

    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8-sig") as annotation_file:
                annotation = json.load(annotation_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"{json_path}: {error}")
            continue

        label = annotation.get("text") if isinstance(annotation, dict) else None
        if not isinstance(label, str):
            errors.append(f"{json_path}: missing or non-string top-level 'text' field")
            continue
        labels.append(label)

    label_counts = Counter(labels)
    duplicate_labels = {
        label: count for label, count in label_counts.items() if count > 1
    }
    return {
        "id": f"{split}/{writer_dir.name}",
        "image_count": image_count,
        "json_count": len(json_paths),
        "valid_label_count": len(labels),
        "labels": set(labels),
        "unique_label_count": len(label_counts),
        "duplicate_labels": duplicate_labels,
        "errors": errors,
    }


def write_report(results: list[dict], output_path: Path) -> None:
    common_labels = set.intersection(*(result["labels"] for result in results)) if results else set()
    total_images = sum(result["image_count"] for result in results)
    total_jsons = sum(result["json_count"] for result in results)
    total_errors = sum(len(result["errors"]) for result in results)

    with output_path.open("w", encoding="utf-8") as report:
        report.write("Korean Word OCR Dataset Label-Duplication Report\n")
        report.write("=" * 72 + "\n\n")
        report.write(f"Writers analyzed: {len(results)}\n")
        report.write(f"Total images: {total_images}\n")
        report.write(f"Total JSON annotations: {total_jsons}\n")
        report.write(f"Unreadable/invalid annotations: {total_errors}\n\n")

        report.write("[Per-writer summary]\n")
        for result in results:
            duplicates = result["duplicate_labels"]
            report.write(
                f"{result['id']}: images={result['image_count']}, "
                f"json={result['json_count']}, valid_labels={result['valid_label_count']}, "
                f"unique_labels={result['unique_label_count']}, "
                f"duplicate_label_types={len(duplicates)}\n"
            )
            if duplicates:
                for label, count in sorted(duplicates.items()):
                    report.write(f"  - {label}\t(count: {count})\n")
            else:
                report.write("  - No duplicate labels within this writer.\n")
            if result["errors"]:
                report.write("  Annotation errors:\n")
                for error in result["errors"]:
                    report.write(f"  - {error}\n")
            report.write("\n")

        report.write("[Labels shared by every writer]\n")
        report.write(f"Common-label count across all {len(results)} writers: {len(common_labels)}\n")
        for label in sorted(common_labels):
            report.write(f"- {label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze image counts and duplicate Korean word labels per writer."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/OCR/ocr_a/다양한 형태의 한글 문자 OCR"),
        help="Directory containing Training/2.단어 and Validation/2.단어.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("korean_ocr_word_analysis.txt"),
        help="Path for the UTF-8 text report.",
    )
    args = parser.parse_args()

    results = []
    for split, word_dir_name in SPLITS:
        split_dir = args.dataset_root / split / word_dir_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Writer directory not found: {split_dir}")
        writer_dirs = sorted(
            (path for path in split_dir.iterdir() if path.is_dir()), key=writer_sort_key
        )
        results.extend(analyze_writer(split, writer_dir) for writer_dir in writer_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, args.output)
    print(f"Analyzed {len(results)} writers. Report saved to: {args.output}")


if __name__ == "__main__":
    main()

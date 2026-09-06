#!/usr/bin/env python3
"""Merge the OCR-B step2 test split into validation without copying image data."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            image_path, label = raw_line.split("\t", 1)
        except ValueError as error:
            raise ValueError(f"Malformed line {line_number} in {path}") from error
        if not image_path.startswith("images/") or not label:
            raise ValueError(f"Invalid record on line {line_number} in {path}")
        records.append((image_path, label))
    return records


def main() -> None:
    root = Path(__file__).resolve().parent / "step2"
    source_images = root.parent.parent / "ocr_b" / "13.한국어글자체" / "01.손글씨" / "1_word"
    training = root / "training"
    validation = root / "validation"
    test = root / "test"
    validation_gt = validation / "gt.txt"
    test_gt = test / "gt.txt"

    if not test.is_dir() or not test_gt.is_file():
        raise RuntimeError("step2/test is missing; there is nothing to merge")

    training_records = parse_manifest(training / "gt.txt")
    validation_records = parse_manifest(validation_gt)
    test_records = parse_manifest(test_gt)

    validation_names = {image_path for image_path, _ in validation_records}
    test_names = {image_path for image_path, _ in test_records}
    if len(validation_names) != len(validation_records) or len(test_names) != len(test_records):
        raise RuntimeError("duplicate image paths found in a manifest")
    if validation_names & test_names:
        raise RuntimeError("validation and test manifests overlap")

    missing_validation = [image_path for image_path, _ in validation_records if not (validation / image_path).is_file()]
    if missing_validation:
        raise RuntimeError(f"validation has {len(missing_validation)} missing images; first: {missing_validation[0]}")

    # A partial local deletion can leave test manifests intact while removing image links.
    # Fall back to the immutable OCR-B source so the merged split remains complete.
    missing_test = [image_path for image_path, _ in test_records if not (test / image_path).is_file()]
    missing_from_source = [image_path for image_path in missing_test if not (source_images / Path(image_path).name).is_file()]
    if missing_from_source:
        raise RuntimeError(f"source OCR-B has {len(missing_from_source)} missing images; first: {missing_from_source[0]}")

    validation_images = validation / "images"
    linked = []
    try:
        for position, (image_path, _) in enumerate(test_records, start=1):
            source = test / image_path
            if not source.is_file():
                source = source_images / Path(image_path).name
            target = validation / image_path
            os.link(source, target)
            linked.append(target)
            if position % 10_000 == 0:
                print(f"linked {position}/{len(test_records)} test images into validation", flush=True)

        combined_records = validation_records + test_records
        descriptor, temporary_name = tempfile.mkstemp(prefix=".gt.txt.", suffix=".tmp", dir=validation)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for image_path, label in combined_records:
                    handle.write(f"{image_path}\t{label}\n")
            os.replace(temporary_name, validation_gt)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

        # Final full validation before deleting the former split.
        final_records = parse_manifest(validation_gt)
        final_names = {image_path for image_path, _ in final_records}
        if len(final_records) != len(combined_records) or len(final_names) != len(combined_records):
            raise RuntimeError("merged validation manifest is incomplete or has duplicate paths")
        if any(not (validation / image_path).is_file() for image_path, _ in final_records):
            raise RuntimeError("merged validation manifest references a missing image")
        if len(training_records) + len(final_records) != 359_997:
            raise RuntimeError("merged split count does not match the source dataset")

        shutil.rmtree(test)
    except BaseException:
        for target in linked:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise

    print(f"training={len(training_records)}")
    print(f"validation={len(final_records)}")
    print(f"recovered_test_images_from_source={len(missing_test)}")
    print("test split removed")


if __name__ == "__main__":
    main()

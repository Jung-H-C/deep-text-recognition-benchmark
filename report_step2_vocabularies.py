#!/usr/bin/env python3
"""Write the sorted unique character sets of every step2 split to a text report."""

from pathlib import Path


def characters_from(manifest: Path) -> set[str]:
    characters = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n\r")
            try:
                _, label = line.split("\t", 1)
            except ValueError as error:
                raise ValueError(f"Malformed line {line_number} in {manifest}") from error
            characters.update(label)
    return characters


def main() -> None:
    root = Path(__file__).resolve().parent / "step2"
    output = root / "label_character_set_by_split.txt"
    character_sets = {
        split: characters_from(root / split / "gt.txt")
        for split in ("training", "validation", "test")
    }

    with output.open("w", encoding="utf-8", newline="\n") as report:
        report.write("OCR-B Step2 split label character sets\n")
        report.write("Each label is decomposed into individual Unicode characters.\n")
        report.write("Characters are sorted in Unicode code-point order.\n\n")
        for split, characters in character_sets.items():
            report.write(f"[{split}]\n")
            report.write(f"unique_character_count: {len(characters)}\n")
            report.write("characters:\n")
            for character in sorted(characters):
                report.write(f"{character}\n")
            report.write("\n")

    for split, characters in character_sets.items():
        print(f"{split}: {len(characters)} unique characters")
    print(f"report: {output}")


if __name__ == "__main__":
    main()

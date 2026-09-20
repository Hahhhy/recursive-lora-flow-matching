#!/usr/bin/env python3
"""Convert COCO captions into deterministic image/caption JSONL pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    filenames = {int(item["id"]): item["file_name"] for item in payload["images"]}
    # One deterministic caption per image prevents five near-duplicate examples
    # from dominating a short idea-validation run.
    first_caption: dict[int, str] = {}
    for item in sorted(payload["annotations"], key=lambda value: int(value["id"])):
        first_caption.setdefault(int(item["image_id"]), str(item["caption"]).strip())
    rows = [
        {"image": filenames[image_id], "caption": caption, "image_id": image_id}
        for image_id, caption in first_caption.items()
        if image_id in filenames and (args.image_dir / filenames[image_id]).is_file()
    ]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    if len(rows) < args.limit:
        raise RuntimeError(f"requested {args.limit} pairs but found only {len(rows)} images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(rows), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()

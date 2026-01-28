#!/usr/bin/env python3
"""
Compose a 2-column grid of score plots from existing PNGs.
Left: Borda, Kemeny, Copeland. Right: Bradley-Terry, Plackett-Luce, Mallows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    new_height = int(round(img.height * width / img.width))
    return img.resize((width, new_height), Image.LANCZOS)


def load_images(paths: List[Path]) -> List[Image.Image]:
    images = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        images.append(Image.open(path).convert("RGB"))
    return images


def compose_grid(
    left: List[Image.Image],
    right: List[Image.Image],
    output: Path,
    gutter: int,
    background: str,
) -> None:
    if len(left) != len(right):
        raise ValueError("Left and right columns must have the same number of images.")
    if not left:
        raise ValueError("No images provided.")

    target_width = max(max(img.width for img in left), max(img.width for img in right))
    left_resized = [resize_to_width(img, target_width) for img in left]
    right_resized = [resize_to_width(img, target_width) for img in right]

    row_heights = [max(l.height, r.height) for l, r in zip(left_resized, right_resized)]
    total_height = sum(row_heights) + gutter * (len(row_heights) + 1)
    total_width = target_width * 2 + gutter * 3

    canvas = Image.new("RGB", (total_width, total_height), color=background)
    y = gutter
    for row_idx, (l_img, r_img, row_h) in enumerate(zip(left_resized, right_resized, row_heights)):
        left_x = gutter
        right_x = gutter * 2 + target_width
        left_y = y + (row_h - l_img.height) // 2
        right_y = y + (row_h - r_img.height) // 2
        canvas.paste(l_img, (left_x, left_y))
        canvas.paste(r_img, (right_x, right_y))
        y += row_h + gutter

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a 2-column grid of score plots.")
    parser.add_argument(
        "--left",
        nargs="+",
        type=Path,
        default=[
            Path("visualization/output/borda_scores.png"),
            Path("visualization/output/kemeny_ranking.png"),
            Path("visualization/output/copeland_scores.png"),
        ],
        help="Left column image paths (top to bottom).",
    )
    parser.add_argument(
        "--right",
        nargs="+",
        type=Path,
        default=[
            Path("visualization/output/bradley_terry_scores.png"),
            Path("visualization/output/plackett_luce_scores.png"),
            Path("visualization/output/mallows_consensus.png"),
        ],
        help="Right column image paths (top to bottom).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/output/score_grid.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--gutter", type=int, default=24)
    parser.add_argument("--background", default="#FFFFFF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_imgs = load_images(args.left)
    right_imgs = load_images(args.right)
    compose_grid(left_imgs, right_imgs, args.output, args.gutter, args.background)
    print(f"Wrote grid to {args.output}")


if __name__ == "__main__":
    main()

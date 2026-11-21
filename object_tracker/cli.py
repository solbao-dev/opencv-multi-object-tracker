"""명령줄 입력을 검증하고 애플리케이션에 전달합니다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .app import AppConfig, run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MOG2 + Greedy tracking + ORB (no deep learning)")
    result.add_argument("--source", default="0", help="Webcam index (0) or video path")
    result.add_argument("--target", type=Path, default=Path("data/target.png"))
    result.add_argument("--output", type=Path, default=Path("outputs"))
    result.add_argument("--learning-rate", type=float, default=0.01)
    result.add_argument("--min-area", type=float, default=400)
    result.add_argument("--max-distance", type=float, default=70)
    result.add_argument("--max-missed", type=int, default=15)
    result.add_argument("--warmup-frames", type=int, default=30)
    result.add_argument("--width", type=int, default=960)
    result.add_argument("--headless", action="store_true", help="Process without GUI")
    result.add_argument("--save-video", action="store_true")
    result.add_argument("--show-mask", action="store_true")
    result.add_argument("--max-frames", type=int)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if args.width < 320 or (args.max_frames is not None and args.max_frames < 1):
        command.error("width must be >= 320 and max-frames must be >= 1")
    args.source = int(args.source) if args.source.isdecimal() else args.source
    try:
        directory = run(AppConfig(**vars(args)))
    except (ValueError, OSError, cv2.error) as error:
        command.exit(1, "Error: {}\n".format(error))
    except KeyboardInterrupt:
        return 130
    print("Saved:", directory)
    return 0

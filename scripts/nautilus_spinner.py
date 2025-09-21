#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
import time
from typing import Sequence, Tuple

# Existing Nautilus logo lifted from crates/common/src/logging/headers.rs
ART_LINES: Tuple[str, ...] = (
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣶⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣾⣿⣿⣿⠀⢸⣿⣿⣿⣿⣶⣶⣤⣀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⢀⣴⡇⢀⣾⣿⣿⣿⣿⣿⠀⣾⣿⣿⣿⣿⣿⣿⣿⠿⠓⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⣰⣿⣿⡀⢸⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⠟⠁⣠⣄⠀⠀⠀⠀",
    "⠀⠀⠀⠀⢠⣿⣿⣿⣇⠀⢿⣿⣿⣿⣿⣿⠀⢻⣿⣿⣿⡿⢃⣠⣾⣿⣿⣧⡀⠀⠀",
    "⠀⠀⠀⠠⣾⣿⣿⣿⣿⣿⣧⠈⠋⢀⣴⣧⠀⣿⡏⢠⡀⢸⣿⣿⣿⣿⣿⣿⣿⡇⠀",
    "⠀⠀⠀⣀⠙⢿⣿⣿⣿⣿⣿⠇⢠⣿⣿⣿⡄⠹⠃⠼⠃⠈⠉⠛⠛⠛⠛⠛⠻⠇⠀",
    "⠀⠀⢸⡟⢠⣤⠉⠛⠿⢿⣿⠀⢸⣿⡿⠋⣠⣤⣄⠀⣾⣿⣿⣶⣶⣶⣦⡄⠀⠀⠀",
    "⠀⠀⠸⠀⣾⠏⣸⣷⠂⣠⣤⠀⠘⢁⣴⣾⣿⣿⣿⡆⠘⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠛⠀⣿⡟⠀⢻⣿⡄⠸⣿⣿⣿⣿⣿⣿⣿⡀⠘⣿⣿⣿⣿⠟⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⢻⡿⠀⠈⠻⣿⣿⣿⣿⣿⡇⠀⢹⣿⠿⠋⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⡘⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀",
)

BLANK_CHARS = {" ", "\u2800"}
ART_HEIGHT = len(ART_LINES)
ART_WIDTH = max(len(line) for line in ART_LINES)
CENTER_X = (ART_WIDTH - 1) / 2
CENTER_Y = (ART_HEIGHT - 1) / 2
RELATIVE_POINTS: Tuple[Tuple[float, float, str], ...] = tuple(
    (float(x) - CENTER_X, float(y) - CENTER_Y, ch)
    for y, line in enumerate(ART_LINES)
    for x, ch in enumerate(line.ljust(ART_WIDTH))
    if ch not in BLANK_CHARS
)

COLOR_PALETTE = (
    (120, 172, 255),
    (88, 200, 255),
    (84, 239, 209),
    (178, 174, 255),
    (255, 196, 247),
)
RESET = "\033[0m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR = "\033[H\033[2J"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a rotating Nautilus logo in the terminal.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=18.0,
        help="Frames per second to play (default: 18)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Number of rotation steps per revolution (default: 24)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Run for N seconds instead of looping indefinitely.",
    )
    parser.add_argument("--once", action="store_true", help="Play a single revolution and exit.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color accents.")
    parser.add_argument(
        "--mode",
        choices=("gyro", "flat"),
        default="gyro",
        help="Animation mode: gyroscopic wobble (default) or flat 2D spin.",
    )
    return parser.parse_args()


def build_frames(step_count: int, mode: str) -> Sequence[Tuple[str, ...]]:
    frames: list[Tuple[str, ...]] = []
    total_steps = max(1, step_count)

    for step in range(total_steps):
        if mode == "flat":
            points = _rotate_flat(step, total_steps)
        else:
            points = _rotate_gyro(step, total_steps)
        frames.append(_points_to_frame(points))

    return tuple(frames)


def _points_to_frame(points: Sequence[Tuple[int, int, str]]) -> Tuple[str, ...]:
    if not points:
        return ("",)

    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y, ch in points:
        gx = x - min_x
        gy = y - min_y
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = ch

    lines = ["".join(row).rstrip() for row in grid]
    return trim_blank_rows(lines)


def _rotate_flat(step: int, step_count: int) -> Sequence[Tuple[int, int, str]]:
    angle = 2.0 * math.pi * (step / step_count)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated: list[Tuple[int, int, str]] = []
    for rel_x, rel_y, ch in RELATIVE_POINTS:
        x = rel_x * cos_a - rel_y * sin_a
        y = rel_x * sin_a + rel_y * cos_a
        rotated.append((int(round(x)), int(round(y)), ch))
    return rotated


def _rotate_gyro(step: int, step_count: int) -> Sequence[Tuple[int, int, str]]:
    t = step / step_count
    theta = 2.0 * math.pi * t
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    squash_bias = 0.20  # squeeze width without over-crushing edge-on frames
    lift_bias = 0.08  # hint at raised relief while keeping internals stable
    global_scale = 1.04

    rotated: list[Tuple[int, int, str]] = []
    for rel_x, rel_y, ch in RELATIVE_POINTS:
        x = rel_x * cos_theta
        width_scale = 1.0 - squash_bias * (1.0 - abs(cos_theta))
        height_shift = rel_x * sin_theta * lift_bias * abs(cos_theta)
        y = (rel_y + height_shift) * width_scale

        x_proj = x * global_scale
        y_proj = y * global_scale

        rotated.append((int(round(x_proj)), int(round(y_proj)), ch))

    return rotated


def trim_blank_rows(lines: Sequence[str]) -> Tuple[str, ...]:
    def is_blank(line: str) -> bool:
        return all(ch in BLANK_CHARS for ch in line) or not line

    start = 0
    end = len(lines)

    while start < end and is_blank(lines[start]):
        start += 1
    while end > start and is_blank(lines[end - 1]):
        end -= 1

    return tuple(lines[start:end]) if start < end else ("",)


def center_frame(frame: Sequence[str], columns: int, lines: int) -> str:
    if not frame:
        return ""

    content_height = len(frame)
    content_width = max(len(row) for row in frame)

    pad_top = max(0, (lines - content_height) // 2)
    pad_left = max(0, (columns - content_width) // 2)
    left_pad = " " * pad_left

    top_padding = "\n" * pad_top if pad_top else ""
    centered_body = "\n".join(left_pad + row for row in frame)

    return top_padding + centered_body


def animate(
    frames: Sequence[Tuple[str, ...]],
    fps: float,
    duration: float | None,
    play_once: bool,
    use_color: bool,
) -> None:
    interval = 1.0 / fps if fps > 0 else 0.0
    colour_cycle = itertools.cycle(COLOR_PALETTE)
    total_frames = len(frames)
    max_frames = total_frames if play_once else None

    sys.stdout.write(HIDE_CURSOR)
    start_time = time.perf_counter()

    try:
        for index, frame in enumerate(itertools.cycle(frames)):
            if max_frames is not None and index >= max_frames:
                break
            if duration is not None and time.perf_counter() - start_time >= duration:
                break

            cols, rows = shutil.get_terminal_size(fallback=(80, 24))
            colour = next(colour_cycle) if use_color else None

            frame_start = time.perf_counter()
            sys.stdout.write(CLEAR)
            if colour is not None:
                r, g, b = colour
                sys.stdout.write(f"\033[38;2;{r};{g};{b}m")
            sys.stdout.write(center_frame(frame, cols, rows))
            sys.stdout.write("\n")
            if colour is not None:
                sys.stdout.write(RESET)
            sys.stdout.flush()

            elapsed = time.perf_counter() - frame_start
            sleep_for = max(0.0, interval - elapsed)
            time.sleep(sleep_for)
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    args = parse_args()
    frames = build_frames(max(4, args.steps), args.mode)
    animate(
        frames=frames,
        fps=max(args.fps, 1e-3),
        duration=args.duration,
        play_once=args.once,
        use_color=not args.no_color,
    )


if __name__ == "__main__":
    main()

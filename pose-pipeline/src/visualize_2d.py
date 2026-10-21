#!/usr/bin/env python3
"""
2D Plane Coordinate Animation Visualizer & Exporter
Part of SLSL Avatar Pipeline

Renders 33-point MediaPipe Pose landmarks across frames on a 2D plane with:
- 3 distinct color groups:
    1. Head / Face (0-10): Neon Cyan
    2. Body / Torso & Limbs (11-14, 23-32): Emerald Green
    3. Hands (15-22): Vivid Coral / Magenta
- Skeletal bones connecting anatomical joints
- Coordinate telemetry HUD (frame, time, wrist velocities)
- Motion trajectory trails for wrists/hands
- Exports to animated GIF or MP4 video
"""

import ast
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch


# MediaPipe 33 Landmark Partitioning
HEAD_INDICES = list(range(0, 11))
BODY_INDICES = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
HANDS_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]

# Distinct 3 Colors
COLOR_HEAD = "#00E5FF"   # Neon Cyan
COLOR_BODY = "#00E676"   # Emerald Green
COLOR_HANDS = "#FF3366"  # Coral / Electric Magenta
COLOR_BONE_HEAD = "#00B4D8"
COLOR_BONE_BODY = "#2E7D32"
COLOR_BONE_HANDS = "#C2185B"
COLOR_TRAIL_L = "#FF80AB"
COLOR_TRAIL_R = "#FF4081"

# Anatomical Connections
HEAD_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

BODY_CONNECTIONS = [
    (11, 12),           # Shoulders
    (11, 13), (12, 14), # Upper arms (shoulder -> elbow)
    (11, 23), (12, 24), # Torso sides
    (23, 24),           # Hips
    (23, 25), (24, 26), # Upper legs (hip -> knee)
    (25, 27), (26, 28), # Lower legs (knee -> ankle)
    (27, 29), (28, 30), # Ankle -> heel
    (29, 31), (30, 32), # Heel -> toe
]

HAND_CONNECTIONS = [
    (13, 15), (14, 16), # Forearm (elbow -> wrist)
    (15, 17), (16, 18), # Wrist -> pinky
    (15, 19), (16, 20), # Wrist -> index
    (15, 21), (16, 22), # Wrist -> thumb
    (17, 19), (18, 20), # Pinky -> index base
]


def load_landmarks_from_csv(csv_path: Path) -> np.ndarray:
    """
    Parses MediaPipe Pose CSV containing 33 landmark columns per row.
    Returns array of shape (frames, 33, 4) with [x, y, z, visibility].
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    frames = []
    for row in rows:
        if len(row) < 33:
            continue
        frame_landmarks = [ast.literal_eval(c) for c in row[:33]]
        frames.append(frame_landmarks)

    return np.array(frames, dtype=np.float32)


def render_2d_animation(
    csv_path: Path,
    out_path: Path,
    fps: int = 15,
    trail_length: int = 10,
    max_frames: int = None,
    dpi: int = 100,
):
    """
    Generates and saves a 2D coordinate animation from the MediaPipe Pose CSV.
    """
    landmarks = load_landmarks_from_csv(csv_path)
    total_frames = len(landmarks)
    if max_frames and max_frames < total_frames:
        landmarks = landmarks[:max_frames]
        total_frames = max_frames

    print(f"Loaded {total_frames} frames from {csv_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up dark-mode scientific canvas
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0B0F19")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)  # Flip y so 0 is at top, matching image coordinates
    ax.set_aspect("equal")

    # Styling grid
    ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("X Coordinate (Normalized [0, 1])", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Y Coordinate (Normalized [0, 1])", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#64748B", labelsize=9)

    # Title & Legend
    title_text = ax.text(
        0.5, -0.01,
        f"2D Pose Coordinate Animation: {csv_path.stem}",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color="#F8FAFC",
        transform=ax.transAxes,
    )

    # Anatomical Category Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="Head / Face (0-10)", markerfacecolor=COLOR_HEAD, markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Body & Limbs (11-14, 23-32)", markerfacecolor=COLOR_BODY, markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Hands & Wrists (15-22)", markerfacecolor=COLOR_HANDS, markersize=8),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#1E293B",
        edgecolor="#334155",
        labelcolor="#E2E8F0",
        fontsize=8.5,
    )

    # Skeletal Bone Line Artists
    head_lines = [ax.plot([], [], color=COLOR_BONE_HEAD, lw=2.2, alpha=0.9)[0] for _ in HEAD_CONNECTIONS]
    body_lines = [ax.plot([], [], color=COLOR_BONE_BODY, lw=2.5, alpha=0.9)[0] for _ in BODY_CONNECTIONS]
    hand_lines = [ax.plot([], [], color=COLOR_BONE_HANDS, lw=2.5, alpha=0.9)[0] for _ in HAND_CONNECTIONS]

    # Landmark Point Scatter Artists
    scatter_head = ax.scatter([], [], s=45, color=COLOR_HEAD, edgecolors="#FFFFFF", linewidths=0.8, zorder=5)
    scatter_body = ax.scatter([], [], s=55, color=COLOR_BODY, edgecolors="#FFFFFF", linewidths=0.8, zorder=5)
    scatter_hands = ax.scatter([], [], s=75, color=COLOR_HANDS, edgecolors="#FFFFFF", linewidths=1.2, zorder=6)

    # Motion Trajectory Trails for Wrists (Left: 15, Right: 16)
    trail_left, = ax.plot([], [], color=COLOR_TRAIL_L, lw=1.8, linestyle=":", alpha=0.75, zorder=4)
    trail_right, = ax.plot([], [], color=COLOR_TRAIL_R, lw=2.0, linestyle="-", alpha=0.85, zorder=4)

    # Telemetry HUD text
    telemetry_text = ax.text(
        0.03, 0.03, "",
        transform=ax.transAxes,
        fontsize=8.5,
        family="monospace",
        color="#38BDF8",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#020617", edgecolor="#1E293B", alpha=0.85),
    )

    def init():
        for line in head_lines + body_lines + hand_lines:
            line.set_data([], [])
        scatter_head.set_offsets(np.empty((0, 2)))
        scatter_body.set_offsets(np.empty((0, 2)))
        scatter_hands.set_offsets(np.empty((0, 2)))
        trail_left.set_data([], [])
        trail_right.set_data([], [])
        telemetry_text.set_text("")
        return head_lines + body_lines + hand_lines + [scatter_head, scatter_body, scatter_hands, trail_left, trail_right, telemetry_text]

    def update(frame_idx):
        pts = landmarks[frame_idx]  # (33, 4)
        xy = pts[:, :2]

        # 1. Update lines
        for line, (i, j) in zip(head_lines, HEAD_CONNECTIONS):
            line.set_data([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]])

        for line, (i, j) in zip(body_lines, BODY_CONNECTIONS):
            line.set_data([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]])

        for line, (i, j) in zip(hand_lines, HAND_CONNECTIONS):
            line.set_data([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]])

        # 2. Update points
        scatter_head.set_offsets(xy[HEAD_INDICES])
        scatter_body.set_offsets(xy[BODY_INDICES])
        scatter_hands.set_offsets(xy[HANDS_INDICES])

        # 3. Update motion trails for left wrist (15) and right wrist (16)
        start_trail = max(0, frame_idx - trail_length)
        left_trail_pts = landmarks[start_trail : frame_idx + 1, 15, :2]
        right_trail_pts = landmarks[start_trail : frame_idx + 1, 16, :2]
        trail_left.set_data(left_trail_pts[:, 0], left_trail_pts[:, 1])
        trail_right.set_data(right_trail_pts[:, 0], right_trail_pts[:, 1])

        # 4. Telemetry info
        rw = pts[16]  # Right wrist
        lw = pts[15]  # Left wrist
        nose = pts[0]
        
        # Velocity if not first frame
        if frame_idx > 0:
            prev_rw = landmarks[frame_idx - 1, 16, :2]
            vel_rw = np.linalg.norm(rw[:2] - prev_rw) * fps
        else:
            vel_rw = 0.0

        telemetry_text.set_text(
            f"Frame: {frame_idx + 1:03d} / {total_frames:03d}  |  t = {(frame_idx / fps):.2f}s\n"
            f"R-Wrist (16): X={rw[0]:.3f} Y={rw[1]:.3f} Z={rw[2]:.3f} (Vel: {vel_rw:.2f}/s)\n"
            f"L-Wrist (15): X={lw[0]:.3f} Y={lw[1]:.3f} Z={lw[2]:.3f}\n"
            f"Nose    (00): X={nose[0]:.3f} Y={nose[1]:.3f}"
        )

        return head_lines + body_lines + hand_lines + [scatter_head, scatter_body, scatter_hands, trail_left, trail_right, telemetry_text]

    print(f"Rendering animation to {out_path} ...")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=False,
        interval=1000 // fps,
    )

    if out_path.suffix.lower() == ".gif":
        ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    elif out_path.suffix.lower() == ".mp4":
        ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        # Default to gif
        out_path = out_path.with_suffix(".gif")
        ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)

    plt.close(fig)
    print(f"Successfully saved 2D coordinate animation: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render 2D Pose Coordinate Animation from MediaPipe CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--out", type=str, default="data/inspection_plots/pose_2d_animation.gif", help="Output .gif or .mp4 path")
    parser.add_argument("--fps", type=int, default=12, help="Animation playback FPS")
    parser.add_argument("--trail", type=int, default=12, help="Wrist trajectory trail length in frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to render")
    args = parser.parse_args()

    render_2d_animation(
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        fps=args.fps,
        trail_length=args.trail,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

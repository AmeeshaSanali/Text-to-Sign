#!/usr/bin/env python3
"""
Scientific Publication-Quality Landmark Coordinate Trajectory Plotter
Part of SLSL Avatar & Gesture Kinematics Pipeline

Plots 33-point MediaPipe Pose landmark coordinate values against Time:
- Panel 1: X(t) Coordinate Trajectories (Head: Cyan, Body: Green, Hands: Pink)
- Panel 2: Y(t) Coordinate Trajectories (Head: Cyan, Body: Green, Hands: Pink)
- Panel 3: Kinematic Velocity v(t) & Gesture Dynamics
Exports high-resolution figures (PNG at 300 DPI, PDF, SVG) for research papers and thesis reports.
"""

import ast
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


# MediaPipe 33 Landmark Partitioning
HEAD_INDICES = list(range(0, 11))
BODY_INDICES = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
HANDS_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]

# Standard 3 Colors
COLOR_HEAD = "#00E5FF"   # Neon Cyan
COLOR_BODY = "#00E676"   # Emerald Green
COLOR_HANDS = "#FF3366"  # Coral Pink / Electric Magenta
COLOR_HEAD_ALT = "#0091EA"
COLOR_BODY_ALT = "#1B5E20"
COLOR_HANDS_ALT = "#D81B60"

# Landmark Labels for Key Articulators
KEY_LANDMARKS = {
    0: ("Nose (Head)", COLOR_HEAD, "-", 2.0),
    11: ("Left Shoulder (Body)", COLOR_BODY, "--", 1.5),
    12: ("Right Shoulder (Body)", COLOR_BODY, "-", 1.5),
    13: ("Left Elbow (Body)", "#69F0AE", ":", 1.2),
    14: ("Right Elbow (Body)", "#00C853", "-.", 1.2),
    15: ("Left Wrist (Hands)", "#FF80AB", "--", 2.2),
    16: ("Right Wrist (Hands)", COLOR_HANDS, "-", 2.5),
    19: ("Left Index (Hands)", "#FF4081", ":", 1.2),
    20: ("Right Index (Hands)", "#C2185B", "-.", 1.2),
}


def load_landmarks(csv_path: Path) -> np.ndarray:
    """Parses MediaPipe Pose CSV containing 33 landmark columns. Shape: (T, 33, 4)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    frames = []
    for row in rows:
        if len(row) < 33:
            continue
        frames.append([ast.literal_eval(c) for c in row[:33]])

    return np.array(frames, dtype=np.float32)


def compute_kinematics(landmarks: np.ndarray, fps: float = 10.0) -> Dict[str, np.ndarray]:
    """
    Computes time-series kinematics:
    - time: t in seconds
    - velocity: ||p(t) - p(t-1)|| / dt
    - acceleration: ||v(t) - v(t-1)|| / dt
    """
    T = len(landmarks)
    dt = 1.0 / fps
    time_sec = np.linspace(0, (T - 1) * dt, T)

    # Coordinates
    coords = landmarks[:, :, :3]  # (T, 33, 3)

    # Velocity: Euclidean displacement per frame / dt
    velocities = np.zeros((T, 33), dtype=np.float32)
    if T > 1:
        diffs = np.diff(coords, axis=0)  # (T-1, 33, 3)
        speeds = np.linalg.norm(diffs, axis=-1) / dt  # (T-1, 33)
        velocities[1:] = speeds
        velocities[0] = velocities[1]

    # Accelerations
    accelerations = np.zeros((T, 33), dtype=np.float32)
    if T > 2:
        v_diffs = np.diff(velocities, axis=0)
        accels = np.abs(v_diffs) / dt
        accelerations[1:] = accels
        accelerations[0] = accelerations[1]

    return {
        "time": time_sec,
        "coords": coords,
        "velocities": velocities,
        "accelerations": accelerations,
        "fps": fps,
    }


def plot_coordinate_trajectories(
    csv_path: Path,
    out_path: Path,
    title: Optional[str] = None,
    fps: float = 10.0,
    dpi: int = 300,
    invert_y: bool = True,
    dark_mode: bool = True,
) -> Path:
    """
    Generates a scientific multi-panel Cartesian coordinate trajectory plot:
    - Subplot 1: X(t) Coordinate Trajectories
    - Subplot 2: Y(t) Coordinate Trajectories
    - Subplot 3: Kinematic Velocity v(t) & Motion Phases
    """
    landmarks = load_landmarks(csv_path)
    kin = compute_kinematics(landmarks, fps=fps)
    t = kin["time"]
    coords = kin["coords"]
    vel = kin["velocities"]
    T = len(t)

    # Invert Y if requested (MediaPipe Y is 0 at top, 1 at bottom; inverted matches standard Cartesian)
    y_coords = 1.0 - coords[:, :, 1] if invert_y else coords[:, :, 1]
    y_label_suffix = " (Cartesian: 1 = Top)" if invert_y else " (MediaPipe: 0 = Top)"

    # Style Configuration
    if dark_mode:
        plt.style.use("dark_background")
        bg_fig = "#0b0f19"
        bg_ax = "#111827"
        text_color = "#f9fafb"
        grid_color = "#374151"
    else:
        plt.style.use("default")
        bg_fig = "#ffffff"
        bg_ax = "#f8fafc"
        text_color = "#0f172a"
        grid_color = "#e2e8f0"

    fig, (ax_x, ax_y, ax_v) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=dpi)
    fig.patch.set_facecolor(bg_fig)

    display_title = title or f"Kinematic Landmark Coordinate Trajectory Analysis: {csv_path.stem}"
    fig.suptitle(display_title, fontsize=14, fontweight="bold", color=text_color, y=0.98)

    # ----------------------------------------------------
    # Subplot 1: X(t) Coordinates
    # ----------------------------------------------------
    ax_x.set_facecolor(bg_ax)
    ax_x.set_ylabel("X Coordinate (Normalized)", fontsize=11, fontweight="600", color=text_color)
    ax_x.grid(True, linestyle="--", alpha=0.5, color=grid_color)
    ax_x.set_ylim(-0.05, 1.05)

    # Background envelope shading for Head, Body, Hands
    head_x = coords[:, HEAD_INDICES, 0]
    body_x = coords[:, BODY_INDICES, 0]
    hands_x = coords[:, HANDS_INDICES, 0]

    ax_x.fill_between(t, head_x.min(axis=1), head_x.max(axis=1), color=COLOR_HEAD, alpha=0.10, label="Head Envelope (0-10)")
    ax_x.fill_between(t, body_x.min(axis=1), body_x.max(axis=1), color=COLOR_BODY, alpha=0.08, label="Body Envelope (11-14, 23-32)")
    ax_x.fill_between(t, hands_x.min(axis=1), hands_x.max(axis=1), color=COLOR_HANDS, alpha=0.12, label="Hands Envelope (15-22)")

    # Plot Key Individual Trajectories
    for idx, (label, color, ls, lw) in KEY_LANDMARKS.items():
        ax_x.plot(t, coords[:, idx, 0], color=color, linestyle=ls, linewidth=lw, label=label)

    ax_x.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.8)
    ax_x.text(0.01, 0.90, "Panel A: Horizontal Motion X(t)", transform=ax_x.transAxes,
              fontsize=10, fontweight="bold", color=text_color,
              bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_ax, edgecolor=grid_color, alpha=0.8))

    # ----------------------------------------------------
    # Subplot 2: Y(t) Coordinates
    # ----------------------------------------------------
    ax_y.set_facecolor(bg_ax)
    ax_y.set_ylabel(f"Y Coordinate{y_label_suffix}", fontsize=11, fontweight="600", color=text_color)
    ax_y.grid(True, linestyle="--", alpha=0.5, color=grid_color)
    ax_y.set_ylim(-0.05, 1.05)

    head_y = y_coords[:, HEAD_INDICES]
    body_y = y_coords[:, BODY_INDICES]
    hands_y = y_coords[:, HANDS_INDICES]

    ax_y.fill_between(t, head_y.min(axis=1), head_y.max(axis=1), color=COLOR_HEAD, alpha=0.10)
    ax_y.fill_between(t, body_y.min(axis=1), body_y.max(axis=1), color=COLOR_BODY, alpha=0.08)
    ax_y.fill_between(t, hands_y.min(axis=1), hands_y.max(axis=1), color=COLOR_HANDS, alpha=0.12)

    for idx, (label, color, ls, lw) in KEY_LANDMARKS.items():
        ax_y.plot(t, y_coords[:, idx], color=color, linestyle=ls, linewidth=lw)

    ax_y.text(0.01, 0.90, "Panel B: Vertical Motion Y(t)", transform=ax_y.transAxes,
              fontsize=10, fontweight="bold", color=text_color,
              bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_ax, edgecolor=grid_color, alpha=0.8))

    # ----------------------------------------------------
    # Subplot 3: Kinematic Velocity v(t)
    # ----------------------------------------------------
    ax_v.set_facecolor(bg_ax)
    ax_v.set_xlabel("Time (seconds)", fontsize=11, fontweight="600", color=text_color)
    ax_v.set_ylabel("Velocity v(t) [units/s]", fontsize=11, fontweight="600", color=text_color)
    ax_v.grid(True, linestyle="--", alpha=0.5, color=grid_color)

    # Plot velocity for Head (Nose), Body (Right Shoulder), Hands (Left and Right Wrists)
    ax_v.plot(t, vel[:, 0], color=COLOR_HEAD, linestyle="-", linewidth=1.5, label="Head Velocity (Nose)")
    ax_v.plot(t, vel[:, 12], color=COLOR_BODY, linestyle="-", linewidth=1.5, label="Body Velocity (Right Shoulder)")
    ax_v.plot(t, vel[:, 15], color="#FF80AB", linestyle="--", linewidth=1.8, label="Left Wrist Velocity")
    ax_v.plot(t, vel[:, 16], color=COLOR_HANDS, linestyle="-", linewidth=2.2, label="Right Wrist Velocity (Dominant)")

    # Highlight Gesture Apex (Maximum hand velocity)
    max_idx = np.argmax(vel[:, 16])
    max_t = t[max_idx]
    max_v = vel[max_idx, 16]
    ax_v.scatter([max_t], [max_v], color="#FFD600", s=60, zorder=5)
    ax_v.annotate(
        f"Peak Velocity\n({max_v:.2f} u/s at {max_t:.2f}s)",
        xy=(max_t, max_v),
        xytext=(max_t + (t[-1] * 0.05), max_v * 0.95),
        arrowprops=dict(facecolor="#FFD600", arrowstyle="->", lw=1.2),
        fontsize=8,
        color=text_color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor="#FFD600", alpha=0.8),
    )

    ax_v.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.8)
    ax_v.text(0.01, 0.90, "Panel C: Kinematic Velocity v(t) & Dynamic Effort", transform=ax_v.transAxes,
              fontsize=10, fontweight="bold", color=text_color,
              bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_ax, edgecolor=grid_color, alpha=0.8))

    # Format Ticks and Ranges
    ax_x.set_xlim(0, t[-1])
    ax_v.set_xlim(0, t[-1])

    # Footer Telemetry Metadata
    footer_text = (
        f"Frames: {T} | FPS: {fps:.1f} | Duration: {t[-1]:.2f}s | "
        f"Peak Velocity: {max_v:.2f} units/s | Landmark Model: MediaPipe Pose 33 | "
        f"3 Color Schema: Cyan (Head), Green (Body), Pink (Hands)"
    )
    fig.text(0.5, 0.01, footer_text, ha="center", fontsize=8, color="#9ca3af")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot 33-point MediaPipe Pose coordinate trajectories over time.")
    parser.add_argument("--csv", type=str, required=True, help="Path to landmark CSV file")
    parser.add_argument("--out", type=str, default=None, help="Output image file (.png, .pdf, .svg)")
    parser.add_argument("--title", type=str, default=None, help="Custom figure title")
    parser.add_argument("--fps", type=float, default=10.0, help="Recording frame rate (default: 10.0)")
    parser.add_argument("--dpi", type=int, default=300, help="Output image resolution (default: 300 DPI)")
    parser.add_argument("--light", action="store_true", help="Render in publication light mode instead of dark mode")
    parser.add_argument("--no-invert-y", action="store_true", help="Do not invert Y coordinates (keep MediaPipe raw 0=top)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = csv_path.parent / f"{csv_path.stem}_trajectory.png"

    print(f"Generating scientific coordinate trajectory plot for: {csv_path.name}")
    result = plot_coordinate_trajectories(
        csv_path=csv_path,
        out_path=out_path,
        title=args.title,
        fps=args.fps,
        dpi=args.dpi,
        invert_y=not args.no_invert_y,
        dark_mode=not args.light,
    )
    print(f"Saved publication trajectory figure to: {result.resolve()}")


if __name__ == "__main__":
    main()

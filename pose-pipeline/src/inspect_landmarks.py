#!/usr/bin/env python3
"""
Landmark Trajectory Inspection Plotter
Part of Track B - Phase 3: Cleaning

Generates comparison plots of raw (10 FPS) vs cleaned & smoothed (30 FPS)
trajectories for the 10 starter signs.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STARTERS = [
    ("BLUE", "Blue_001"),
    ("TODAY", "Today_036"),
    ("NO", "No_003"),
    ("HELLO", "Hello_049"),
    ("THANK_YOU", "Thank you_058"),
    ("YES", "Yes_002"),
    ("MOTHER", "Mother_027"),
    ("COME", "Come_030"),
    ("EAT", "Eat_021"),
    ("GO", "Go_034"),
]


def generate_comparison_plots(
    raw_dir: Path = Path("data/landmarks_raw"),
    clean_dir: Path = Path("data/landmarks_clean"),
    out_dir: Path = Path("data/inspection_plots"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Generating Before/After Inspection Plots into {out_dir} ===")

    for gloss, take in STARTERS:
        r_file = raw_dir / gloss / f"{take}.npz"
        c_file = clean_dir / gloss / f"{take}.npz"

        if not r_file.exists() or not c_file.exists():
            print(f"Skipping {gloss}: missing raw or clean file.")
            continue

        r_data = np.load(r_file)
        c_data = np.load(c_file)

        r_pw = r_data["pose_world"]
        c_pw = c_data["pose_world"]

        # Select dominant hand wrist (Left 15 vs Right 16)
        r_var_rw = np.var(r_pw[:, 16, :3])
        r_var_lw = np.var(r_pw[:, 15, :3])
        wrist_idx = 16 if r_var_rw >= r_var_lw else 15
        wrist_name = "Right Wrist" if wrist_idx == 16 else "Left Wrist"

        raw_w = r_pw[:, wrist_idx, :3]
        clean_w = c_pw[:, wrist_idx, :3]

        t_raw = np.linspace(0, (len(raw_w) - 1) / 10.0, len(raw_w))
        t_clean = np.linspace(0, (len(clean_w) - 1) / 30.0, len(clean_w))

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axis_labels = ["X (Lateral, m)", "Y (Vertical, m)", "Z (Depth, m)"]

        for i in range(3):
            axes[i].plot(t_raw, raw_w[:, i], "o--", color="crimson", alpha=0.5, label="Raw (10 FPS)")
            axes[i].plot(t_clean, clean_w[:, i], "-", color="royalblue", linewidth=2, label="Cleaned (30 FPS, Smoothed)")
            axes[i].set_ylabel(axis_labels[i])
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(loc="upper right")

        axes[2].set_xlabel("Time (seconds)")
        plt.suptitle(f"{gloss} ({take}) — {wrist_name} Trajectory Comparison", fontsize=13)
        plt.tight_layout()

        plot_path = out_dir / f"{gloss}_comparison.png"
        plt.savefig(plot_path, dpi=140)
        plt.close()
        print(f"Generated: {plot_path.name}")


if __name__ == "__main__":
    generate_comparison_plots()

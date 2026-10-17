#!/usr/bin/env python3
"""
Landmark Cleaning and Normalization Pipeline
Part of Track B - Phase 3: Cleaning

Processes raw 3D MediaPipe landmark archives (.npz):
1. Trims leading and trailing idle resting frames.
2. Interpolates tracking dropouts (gaps <= 5 frames).
3. Applies Savitzky-Golay smoothing.
4. Centers mid-hip at origin (0, 0, 0) and normalizes to standard shoulder width.
5. Resamples from 10.0 FPS to 30.0 FPS via PCHIP cubic interpolation.
6. Supports mirroring for left-handed signers.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal
from scipy.interpolate import PchipInterpolator


# Symmetric pose landmark index pairs for left/right swapping
POSE_SWAP_PAIRS = [
    (1, 4),   # inner eye
    (2, 5),   # eye
    (3, 6),   # outer eye
    (7, 8),   # ear
    (9, 10),  # mouth corner
    (11, 12), # shoulder
    (13, 14), # elbow
    (15, 16), # wrist
    (17, 18), # pinky
    (19, 20), # index
    (21, 22), # thumb
    (23, 24), # hip
    (25, 26), # knee
    (27, 28), # ankle
    (29, 30), # heel
    (31, 32), # foot index
]


def detect_idle_boundaries(pose_world: np.ndarray, fps: float = 10.0, pad_frames: int = 2) -> tuple[int, int]:
    """
    Identifies the active signing segment by detecting wrist velocity and height transitions.
    Trims dead air / resting frames before hand elevation and after lowering.
    """
    n_frames = len(pose_world)
    if n_frames <= 10:
        return 0, n_frames

    # Left and Right wrists (indices 15 and 16)
    lw = pose_world[:, 15, :3]
    rw = pose_world[:, 16, :3]
    mid_hip = (pose_world[:, 23, :3] + pose_world[:, 24, :3]) / 2.0

    # Vertical elevation above mid-hip (MediaPipe world coords: -y is upwards)
    lw_elev = -(lw[:, 1] - mid_hip[:, 1])
    rw_elev = -(rw[:, 1] - mid_hip[:, 1])
    max_elev = np.maximum(lw_elev, rw_elev)

    # Wrist speed (m/s)
    speed = np.zeros(n_frames, dtype=np.float32)
    lw_diff = np.linalg.norm(np.diff(lw, axis=0), axis=1) * fps
    rw_diff = np.linalg.norm(np.diff(rw, axis=0), axis=1) * fps
    speed[1:] = np.maximum(lw_diff, rw_diff)

    # Activity metric: elevated hands (> 0.1m above hips) OR moving hands (> 0.2 m/s)
    active = (max_elev > 0.08) | (speed > 0.20)

    active_indices = np.where(active)[0]
    if len(active_indices) == 0:
        return 0, n_frames

    start_idx = max(0, active_indices[0] - pad_frames)
    end_idx = min(n_frames, active_indices[-1] + pad_frames + 1)

    # Ensure minimum 8 frames remain
    if end_idx - start_idx < 8:
        return 0, n_frames

    return int(start_idx), int(end_idx)


def interpolate_gaps_1d(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly interpolates NaN values in a 1D sequence if gap length <= max_gap."""
    out = arr.copy()
    n = len(arr)
    valid_mask = ~np.isnan(arr)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return out

    # Fill internal gaps
    for k in range(len(valid_idx) - 1):
        idx_prev = valid_idx[k]
        idx_next = valid_idx[k + 1]
        gap_size = idx_next - idx_prev - 1
        if 1 <= gap_size <= max_gap:
            val_prev = arr[idx_prev]
            val_next = arr[idx_next]
            step = (val_next - val_prev) / (gap_size + 1)
            for j in range(1, gap_size + 1):
                out[idx_prev + j] = val_prev + j * step

    # Fill small boundary gaps (<= 3 frames) with nearest value
    if valid_idx[0] > 0 and valid_idx[0] <= 3:
        out[:valid_idx[0]] = arr[valid_idx[0]]
    if n - 1 - valid_idx[-1] > 0 and (n - 1 - valid_idx[-1]) <= 3:
        out[valid_idx[-1] + 1:] = arr[valid_idx[-1]]

    return out


def interpolate_joints(landmarks: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Interpolates NaN gaps for each joint and coordinate axis."""
    out = np.empty_like(landmarks)
    n_frames, n_joints, n_dim = landmarks.shape

    for j in range(n_joints):
        for d in range(n_dim):
            out[:, j, d] = interpolate_gaps_1d(landmarks[:, j, d], max_gap=max_gap)

    return out


def smooth_landmarks(landmarks: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """Applies Savitzky-Golay smoothing on continuous valid segments."""
    n_frames = len(landmarks)
    if n_frames < window_length:
        return landmarks.copy()

    win = window_length if window_length % 2 == 1 else window_length + 1
    if win > n_frames:
        win = n_frames if n_frames % 2 == 1 else n_frames - 1
    if win <= polyorder:
        return landmarks.copy()

    out = landmarks.copy()
    n_joints = landmarks.shape[1]
    n_dim = landmarks.shape[2]

    for j in range(n_joints):
        for d in range(n_dim):
            col = landmarks[:, j, d]
            valid_mask = ~np.isnan(col)
            if np.all(valid_mask):
                out[:, j, d] = scipy.signal.savgol_filter(col, window_length=win, polyorder=polyorder)
            elif np.any(valid_mask):
                # Smooth continuous valid blocks
                valid_indices = np.where(valid_mask)[0]
                blocks = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
                for blk in blocks:
                    if len(blk) >= win:
                        smoothed_blk = scipy.signal.savgol_filter(col[blk], window_length=win, polyorder=polyorder)
                        out[blk, j, d] = smoothed_blk

    return out


def resample_landmarks(landmarks: np.ndarray, in_fps: float, target_fps: float = 30.0) -> np.ndarray:
    """Resamples landmark sequence from in_fps to target_fps using PCHIP interpolation."""
    n_in = len(landmarks)
    if n_in <= 1:
        return landmarks.copy()

    duration = (n_in - 1) / in_fps
    n_out = max(2, int(round(duration * target_fps)) + 1)

    t_in = np.linspace(0.0, duration, n_in)
    t_out = np.linspace(0.0, duration, n_out)

    n_joints = landmarks.shape[1]
    n_dim = landmarks.shape[2]
    out = np.full((n_out, n_joints, n_dim), np.nan, dtype=np.float32)

    for j in range(n_joints):
        for d in range(n_dim):
            col = landmarks[:, j, d]
            valid_mask = ~np.isnan(col)
            if np.sum(valid_mask) >= 2:
                valid_idx = np.where(valid_mask)[0]
                t_sub = t_in[valid_idx]
                val_sub = col[valid_idx]
                pchip = PchipInterpolator(t_sub, val_sub, extrapolate=False)
                out[:, j, d] = pchip(t_out)

    return out


def normalize_skeleton(
    pose_world: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    reference_shoulder_width: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Centers the skeleton so mid-hip is at origin (0, 0, 0)
    and scales upper body to reference shoulder width.
    """
    # Mid-hip position per frame (Left Hip index 23, Right Hip index 24)
    mid_hip = (pose_world[:, 23, :3] + pose_world[:, 24, :3]) / 2.0  # (N, 3)

    # Compute median shoulder width (Left Shoulder 11, Right Shoulder 12)
    shoulder_dist = np.linalg.norm(pose_world[:, 11, :3] - pose_world[:, 12, :3], axis=1)
    median_sw = float(np.nanmedian(shoulder_dist))
    scale = (reference_shoulder_width / median_sw) if median_sw > 0.05 else 1.0

    # Center pose around mid-hip and scale
    norm_pose = pose_world.copy()
    for i in range(33):
        norm_pose[:, i, :3] = (norm_pose[:, i, :3] - mid_hip) * scale

    # Scale hands (preserving wrist-relative coords)
    norm_lh = left_hand.copy() * scale
    norm_rh = right_hand.copy() * scale

    return norm_pose, norm_lh, norm_rh, scale


def mirror_all_landmarks(
    pose_world: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverts X-axis and swaps bilateral limbs to mirror left-handed signs."""
    mirrored_pose = pose_world.copy()
    # Invert X
    mirrored_pose[:, :, 0] *= -1.0

    # Swap paired bilateral joints
    for p_left, p_right in POSE_SWAP_PAIRS:
        temp = mirrored_pose[:, p_left].copy()
        mirrored_pose[:, p_left] = mirrored_pose[:, p_right]
        mirrored_pose[:, p_right] = temp

    # Swap hands and invert their X coordinates
    mirrored_lh = right_hand.copy()
    mirrored_rh = left_hand.copy()
    mirrored_lh[:, :, 0] *= -1.0
    mirrored_rh[:, :, 0] *= -1.0

    return mirrored_pose, mirrored_lh, mirrored_rh


def clean_landmark_data(
    in_npz_path: Path,
    out_npz_path: Path,
    target_fps: float = 30.0,
    trim_idle: bool = True,
    mirrored: bool = False,
) -> dict:
    data = np.load(in_npz_path)
    pose_world = data["pose_world"]       # (N, 33, 4)
    left_hand = data["left_hand_world"]   # (N, 21, 3)
    right_hand = data["right_hand_world"] # (N, 21, 3)
    in_fps = float(data.get("fps", 10.0))
    orig_len = len(pose_world)

    # 1. Idle Frame Trimming
    if trim_idle:
        start_idx, end_idx = detect_idle_boundaries(pose_world, fps=in_fps)
        pose_world = pose_world[start_idx:end_idx]
        left_hand = left_hand[start_idx:end_idx]
        right_hand = right_hand[start_idx:end_idx]
    else:
        start_idx, end_idx = 0, orig_len

    # 2. Gap Interpolation
    pose_world = interpolate_joints(pose_world, max_gap=5)
    left_hand = interpolate_joints(left_hand, max_gap=5)
    right_hand = interpolate_joints(right_hand, max_gap=5)

    # 3. Savitzky-Golay Smoothing
    pose_world = smooth_landmarks(pose_world, window_length=5, polyorder=2)
    left_hand = smooth_landmarks(left_hand, window_length=5, polyorder=2)
    right_hand = smooth_landmarks(right_hand, window_length=5, polyorder=2)

    # 4. Normalization (Mid-hip root & shoulder scale)
    pose_world, left_hand, right_hand, scale = normalize_skeleton(
        pose_world, left_hand, right_hand, reference_shoulder_width=0.35
    )

    # 5. Mirroring (if required)
    if mirrored:
        pose_world, left_hand, right_hand = mirror_all_landmarks(
            pose_world, left_hand, right_hand
        )

    # 6. Resampling to Target FPS (30 FPS)
    resampled_pose = resample_landmarks(pose_world, in_fps=in_fps, target_fps=target_fps)
    resampled_lh = resample_landmarks(left_hand, in_fps=in_fps, target_fps=target_fps)
    resampled_rh = resample_landmarks(right_hand, in_fps=in_fps, target_fps=target_fps)

    # Hand detection mask at target FPS
    lh_detected = ~np.isnan(resampled_lh[:, 0, 0])
    rh_detected = ~np.isnan(resampled_rh[:, 0, 0])
    hand_detected = np.stack([lh_detected, rh_detected], axis=1)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        pose_world=resampled_pose,
        left_hand_world=resampled_lh,
        right_hand_world=resampled_rh,
        hand_detected=hand_detected,
        fps=target_fps,
        frame_count=len(resampled_pose),
        duration_sec=float((len(resampled_pose) - 1) / target_fps),
        trim_range=np.array([start_idx, end_idx, orig_len]),
        scale_factor=scale,
        mirrored=mirrored,
    )

    return {
        "input": str(in_npz_path),
        "output": str(out_npz_path),
        "orig_frames": orig_len,
        "trimmed_frames": end_idx - start_idx,
        "clean_frames": len(resampled_pose),
        "clean_fps": target_fps,
        "duration": round(float((len(resampled_pose) - 1) / target_fps), 2),
        "trim_range": (start_idx, end_idx),
        "scale": round(scale, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean, smooth, and resample 3D landmarks.")
    parser.add_argument("--in", dest="in_file", required=True, help="Input raw .npz file")
    parser.add_argument("--out", dest="out_file", required=True, help="Output clean .npz file")
    parser.add_argument("--fps", default=30.0, type=float, help="Target FPS (default: 30.0)")
    parser.add_argument("--no-trim", action="store_true", help="Disable idle frame trimming")
    parser.add_argument("--mirrored", action="store_true", help="Mirror left/right landmarks")
    args = parser.parse_args()

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    res = clean_landmark_data(
        in_npz_path=in_path,
        out_npz_path=out_path,
        target_fps=args.fps,
        trim_idle=not args.no_trim,
        mirrored=args.mirrored,
    )

    print(f"Cleaning complete for: {in_path.name}")
    print(f"  - Original: {res['orig_frames']} frames @ 10 FPS")
    print(f"  - Trimmed:  {res['trimmed_frames']} frames (range: {res['trim_range']})")
    print(f"  - Output:   {res['clean_frames']} frames @ {res['clean_fps']} FPS ({res['duration']}s)")
    print(f"  - Scale:    {res['scale']}x")
    print(f"Saved to: {res['output']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bone Retargeting Pipeline for Mixamo Y Bot
Part of Track B - Phase 4: Retargeting to Y Bot

Converts cleaned 3D landmark trajectories (at 30 FPS) into local bone rotation
quaternions matching Unity's HumanBodyBones hierarchy.
Exports animations as JSON compatible with PoseJsonLoader.cs and SignPlayer.cs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalizes vector to unit length."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm < eps, np.zeros_like(v), v / np.maximum(norm, eps))


def quat_from_two_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes quaternion [x, y, z, w] that rotates unit vector u to unit vector v.
    Supports single vectors or batched arrays of shape (N, 3).
    """
    u = normalize_vec(u)
    v = normalize_vec(v)

    dot = np.sum(u * v, axis=-1, keepdims=True)
    cross = np.cross(u, v)

    # w = 1.0 + dot
    w = 1.0 + dot
    q = np.concatenate([cross, w], axis=-1)

    # Handle anti-parallel vectors (dot ~= -1)
    anti_mask = (dot < -0.9999).squeeze(-1)
    if np.any(anti_mask):
        # Choose an orthogonal axis
        ortho = np.zeros_like(u)
        ortho[:, 0] = np.where(np.abs(u[:, 0]) < 0.9, 1.0, 0.0)
        ortho[:, 1] = np.where(np.abs(u[:, 0]) >= 0.9, 1.0, 0.0)
        axis = normalize_vec(np.cross(u, ortho))
        q[anti_mask] = np.concatenate([axis[anti_mask], np.zeros((np.sum(anti_mask), 1))], axis=-1)

    return normalize_vec(q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Inverts unit quaternion [x, y, z, w] -> [-x, -y, -z, w]."""
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions q1 * q2."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=-1)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates vector v by quaternion q: q * [v, 0] * q^-1."""
    v_quat = np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1)
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[..., :3]


def continuous_quaternions(quats: np.ndarray) -> np.ndarray:
    """Ensures quaternion continuity (no sign flips between consecutive frames)."""
    out = quats.copy()
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0.0:
            out[t] = -out[t]
    return out


def retarget_arms(pose_world: np.ndarray) -> dict[str, np.ndarray]:
    """
    Retargets upper body arms (UpperArm, LowerArm) to Mixamo Y Bot T-Pose rest directions.
    In T-Pose:
      Right Arm points along +X: (1, 0, 0)
      Left Arm points along -X: (-1, 0, 0)
    Coordinate conversion from MediaPipe World to Unity:
      p_unity = [-x_mp, -y_mp, -z_mp]
    """
    n_frames = len(pose_world)

    # Convert coordinates to Unity World space
    # 11: L Shoulder, 12: R Shoulder
    # 13: L Elbow, 14: R Elbow
    # 15: L Wrist, 16: R Wrist
    ls = -pose_world[:, 11, :3]
    rs = -pose_world[:, 12, :3]
    le = -pose_world[:, 13, :3]
    re = -pose_world[:, 14, :3]
    lw = -pose_world[:, 15, :3]
    rw = -pose_world[:, 16, :3]

    # Target unit direction vectors
    r_upper_dir = normalize_vec(re - rs)
    r_lower_dir = normalize_vec(rw - re)
    l_upper_dir = normalize_vec(le - ls)
    l_lower_dir = normalize_vec(lw - le)

    # Rest directions for Mixamo T-Pose
    v_rest_r = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_frames, 1))
    v_rest_l = np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), (n_frames, 1))

    # --- Right Arm ---
    # Upper Arm: Rest -> Target
    q_r_upper = quat_from_two_vectors(v_rest_r, r_upper_dir)
    q_r_upper = continuous_quaternions(q_r_upper)

    # Lower Arm: transform target direction into parent upper arm's local space
    r_lower_dir_local = quat_rotate_vector(quat_conjugate(q_r_upper), r_lower_dir)
    q_r_lower = quat_from_two_vectors(v_rest_r, r_lower_dir_local)
    q_r_lower = continuous_quaternions(q_r_lower)

    # --- Left Arm ---
    # Upper Arm: Rest -> Target
    q_l_upper = quat_from_two_vectors(v_rest_l, l_upper_dir)
    q_l_upper = continuous_quaternions(q_l_upper)

    # Lower Arm: transform target direction into parent upper arm's local space
    l_lower_dir_local = quat_rotate_vector(quat_conjugate(q_l_upper), l_lower_dir)
    q_l_lower = quat_from_two_vectors(v_rest_l, l_lower_dir_local)
    q_l_lower = continuous_quaternions(q_l_lower)

    return {
        "RightUpperArm": q_r_upper,
        "RightLowerArm": q_r_lower,
        "LeftUpperArm": q_l_upper,
        "LeftLowerArm": q_l_lower,
    }


def retarget_hands_and_fingers(
    left_hand_world: np.ndarray,
    right_hand_world: np.ndarray,
    hand_detected: np.ndarray,
    arm_rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Retargets wrist orientations and finger joint curls.
    If a hand is undetected, outputs identity/rest rotation for that hand.
    """
    n_frames = len(hand_detected)
    rotations = {}

    # Neutral identity rotation for resting/undetected bones: [0, 0, 0, 1]
    identity_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_frames, 1))

    # Finger bones map: (Proximal, Intermediate, Distal)
    finger_indices = {
        "Index": (5, 6, 7, 8),
        "Middle": (9, 10, 11, 12),
        "Ring": (13, 14, 15, 16),
        "Little": (17, 18, 19, 20),
    }

    for side, hand, is_right in [("Right", right_hand_world, True), ("Left", left_hand_world, False)]:
        side_idx = 1 if is_right else 0
        wrist_bone = f"{side}Hand"
        v_rest = np.array([1.0, 0.0, 0.0] if is_right else [-1.0, 0.0, 0.0], dtype=np.float32)

        # Hand wrist orientation
        q_wrist = np.tile(identity_rot[0], (n_frames, 1))

        # Check detected frames
        valid_frames = np.where(hand_detected[:, side_idx])[0]
        if len(valid_frames) > 0:
            # Palm direction: Wrist (0) -> Middle MCP (9)
            palm_vecs = hand[:, 9, :3] - hand[:, 0, :3]
            palm_dir = normalize_vec(palm_vecs)

            # Convert to Unity world space
            palm_dir_unity = -palm_dir

            # Parent arm rotation (Upper * Lower)
            q_parent_arm = quat_multiply(arm_rotations[f"{side}UpperArm"], arm_rotations[f"{side}LowerArm"])
            palm_dir_local = quat_rotate_vector(quat_conjugate(q_parent_arm), palm_dir_unity)

            # Compute wrist rotation
            computed_wrist = quat_from_two_vectors(np.tile(v_rest, (n_frames, 1)), palm_dir_local)
            q_wrist[valid_frames] = continuous_quaternions(computed_wrist)[valid_frames]

        rotations[wrist_bone] = q_wrist

        # Retarget fingers (curl angles)
        for finger_name, (mcp, pip, dip, tip) in finger_indices.items():
            prox_name = f"{side}{finger_name}Proximal"
            inter_name = f"{side}{finger_name}Intermediate"
            dist_name = f"{side}{finger_name}Distal"

            q_prox = np.tile(identity_rot[0], (n_frames, 1))
            q_inter = np.tile(identity_rot[0], (n_frames, 1))
            q_dist = np.tile(identity_rot[0], (n_frames, 1))

            if len(valid_frames) > 0:
                # Segment vectors
                v1 = normalize_vec(hand[:, pip] - hand[:, mcp])
                v2 = normalize_vec(hand[:, dip] - hand[:, pip])
                v3 = normalize_vec(hand[:, tip] - hand[:, dip])

                # Curl angles between consecutive segments
                dot1 = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
                dot2 = np.clip(np.sum(v2 * v3, axis=-1), -1.0, 1.0)

                angle_pip = np.arccos(dot1) # radians
                angle_dip = np.arccos(dot2)

                # Clamp angles to natural limits [0, 80 deg]
                angle_pip = np.clip(angle_pip, 0.0, np.radians(80.0))
                angle_dip = np.clip(angle_dip, 0.0, np.radians(70.0))

                # Local bend axis for fingers in Mixamo (bend along Z)
                half_pip = angle_pip / 2.0
                half_dip = angle_dip / 2.0

                bend_z_pip = np.stack([np.zeros_like(half_pip), np.zeros_like(half_pip), np.sin(half_pip), np.cos(half_pip)], axis=-1)
                bend_z_dip = np.stack([np.zeros_like(half_dip), np.zeros_like(half_dip), np.sin(half_dip), np.cos(half_dip)], axis=-1)

                q_inter[valid_frames] = bend_z_pip[valid_frames]
                q_dist[valid_frames] = bend_z_dip[valid_frames]

            rotations[prox_name] = q_prox
            rotations[inter_name] = q_inter
            rotations[dist_name] = q_dist

        # Thumb
        for thumb_joint in [f"{side}ThumbProximal", f"{side}ThumbIntermediate", f"{side}ThumbDistal"]:
            rotations[thumb_joint] = identity_rot.copy()

    return rotations


def retarget_and_export_json(
    clean_npz_path: Path,
    out_json_path: Path,
    gloss_name: str = None,
) -> dict:
    data = np.load(clean_npz_path)
    pose_world = data["pose_world"]
    left_hand = data["left_hand_world"]
    right_hand = data["right_hand_world"]
    hand_detected = data["hand_detected"]
    fps = float(data.get("fps", 30.0))
    n_frames = len(pose_world)

    gloss = gloss_name or clean_npz_path.parent.name

    # 1. Arm bone rotations
    arm_rotations = retarget_arms(pose_world)

    # 2. Hand and finger rotations
    hand_rotations = retarget_hands_and_fingers(
        left_hand_world=left_hand,
        right_hand_world=right_hand,
        hand_detected=hand_detected,
        arm_rotations=arm_rotations,
    )

    all_rotations = {**arm_rotations, **hand_rotations}

    # Format JSON
    frames_list = []
    for f in range(n_frames):
        frame_bones = {}
        for bone_name, quat_arr in all_rotations.items():
            q = quat_arr[f]
            frame_bones[bone_name] = [
                round(float(q[0]), 5),
                round(float(q[1]), 5),
                round(float(q[2]), 5),
                round(float(q[3]), 5),
            ]
        frames_list.append({"bones": frame_bones})

    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write formatted JSON where each bone quaternion [x, y, z, w] is on a single compact line
    json_lines = [
        "{",
        f'  "gloss": "{gloss}",',
        f'  "language": "slsl",',
        f'  "fps": {int(fps)},',
        f'  "frameCount": {n_frames},',
        f'  "durationSec": {round((n_frames - 1) / fps, 2)},',
        '  "frames": ['
    ]

    for f_idx, frame in enumerate(frames_list):
        json_lines.append("    {")
        json_lines.append('      "bones": {')
        b_lines = [
            f'        "{b_name}": [{b_val[0]}, {b_val[1]}, {b_val[2]}, {b_val[3]}]'
            for b_name, b_val in frame["bones"].items()
        ]
        json_lines.append(",\n".join(b_lines))
        json_lines.append("      }")
        comma = "," if f_idx < n_frames - 1 else ""
        json_lines.append(f"    }}{comma}")

    json_lines.append("  ]")
    json_lines.append("}")

    with open(out_json_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(json_lines))

    return {
        "gloss": gloss,
        "json_path": str(out_json_path),
        "frames": n_frames,
        "fps": fps,
        "duration": round((n_frames - 1) / fps, 2),
        "bone_count": len(all_rotations),
    }


def main():
    parser = argparse.ArgumentParser(description="Retarget 3D landmarks to Unity Humanoid avatar.")
    parser.add_argument("--in", dest="in_file", required=True, help="Input cleaned .npz file")
    parser.add_argument("--out", dest="out_file", required=True, help="Output animation .json file")
    parser.add_argument("--gloss", default=None, help="Sign gloss identifier (e.g. BLUE)")
    args = parser.parse_args()

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()

    res = retarget_and_export_json(in_path, out_path, gloss_name=args.gloss)

    print(f"Retargeting complete for: {res['gloss']}")
    print(f"  - Frames:   {res['frames']} @ {res['fps']} FPS ({res['duration']}s)")
    print(f"  - Bones:    {res['bone_count']} HumanBodyBones")
    print(f"  - Exported: {res['json_path']}")


if __name__ == "__main__":
    main()

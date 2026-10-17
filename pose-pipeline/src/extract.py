#!/usr/bin/env python3
"""
MediaPipe Landmark Extractor (Pose + Hands)
Part of Track B - Phase 2: Extraction with Hands

Extracts 33 pose world landmarks and 21 left/right hand landmarks per frame from video.
Outputs numpy archives (.npz) with NaNs for undetected hands and creates debug overlay videos.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Pose connections
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # upper body / arms
    (11, 23), (12, 24), (23, 24),                      # torso
]

# MediaPipe Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17)                                  # palm base
]


def draw_landmarks(frame, pose_landmarks, left_hand, right_hand):
    """Draws clean skeletons and keypoints on an image frame for debug inspection."""
    h, w, _ = frame.shape
    vis_frame = frame.copy()

    # Draw Pose
    if pose_landmarks is not None:
        coords = {}
        for idx in range(min(len(pose_landmarks), 25)): # focus on upper body
            lm = pose_landmarks[idx]
            if lm[3] > 0.3: # visibility threshold
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                coords[idx] = (cx, cy)
                cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 0), -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in coords and p2 in coords:
                cv2.line(vis_frame, coords[p1], coords[p2], (0, 200, 0), 2)

    # Draw Left Hand (Green/Cyan)
    if left_hand is not None and not np.isnan(left_hand).all():
        coords_lh = {}
        for idx in range(21):
            lx, ly = int(left_hand[idx, 0] * w), int(left_hand[idx, 1] * h)
            coords_lh[idx] = (lx, ly)
            cv2.circle(vis_frame, (lx, ly), 3, (255, 255, 0), -1)
        for p1, p2 in HAND_CONNECTIONS:
            if p1 in coords_lh and p2 in coords_lh:
                cv2.line(vis_frame, coords_lh[p1], coords_lh[p2], (255, 200, 0), 2)

    # Draw Right Hand (Magenta/Pink)
    if right_hand is not None and not np.isnan(right_hand).all():
        coords_rh = {}
        for idx in range(21):
            rx, ry = int(right_hand[idx, 0] * w), int(right_hand[idx, 1] * h)
            coords_rh[idx] = (rx, ry)
            cv2.circle(vis_frame, (rx, ry), 3, (255, 0, 255), -1)
        for p1, p2 in HAND_CONNECTIONS:
            if p1 in coords_rh and p2 in coords_rh:
                cv2.line(vis_frame, coords_rh[p1], coords_rh[p2], (255, 0, 200), 2)

    return vis_frame


def extract_video_landmarks(
    video_path: Path,
    out_path: Path,
    overlay_path: Path = None,
    csv_ref_path: Path = None,
    pose_model_path: Path = Path("data/models/pose_landmarker.task"),
    hand_model_path: Path = Path("data/models/hand_landmarker.task"),
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not pose_model_path.exists():
        raise FileNotFoundError(f"Pose model not found: {pose_model_path}")
    if not hand_model_path.exists():
        raise FileNotFoundError(f"Hand model not found: {hand_model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_pose = python.BaseOptions(model_asset_path=str(pose_model_path))
    opts_pose = vision.PoseLandmarkerOptions(
        base_options=base_pose,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    base_hand = python.BaseOptions(model_asset_path=str(hand_model_path))
    opts_hand = vision.HandLandmarkerOptions(
        base_options=base_hand,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    out_writer = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    pose_world_list = []
    pose_image_list = []
    left_hand_world_list = []
    left_hand_image_list = []
    right_hand_world_list = []
    right_hand_image_list = []
    hand_detected_list = [] # [left_detected, right_detected]

    with vision.PoseLandmarker.create_from_options(opts_pose) as pose_landmarker, \
         vision.HandLandmarker.create_from_options(opts_hand) as hand_landmarker:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000.0 / fps))

            pose_res = pose_landmarker.detect_for_video(mp_img, timestamp_ms)
            hand_res = hand_landmarker.detect_for_video(mp_img, timestamp_ms)

            # 1. Process Pose
            p_world = np.full((33, 4), np.nan, dtype=np.float32)
            p_img = np.full((33, 4), np.nan, dtype=np.float32)
            pose_lw = None
            pose_rw = None

            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                lms = pose_res.pose_landmarks[0]
                wlms = pose_res.pose_world_landmarks[0]
                for i in range(min(33, len(lms))):
                    p_img[i] = [lms[i].x, lms[i].y, lms[i].z, lms[i].visibility]
                    p_world[i] = [wlms[i].x, wlms[i].y, wlms[i].z, wlms[i].visibility]
                pose_lw = (p_img[15, 0], p_img[15, 1])
                pose_rw = (p_img[16, 0], p_img[16, 1])

            # 2. Process Hands (Associating detected hands by proximity to pose wrists)
            lh_world = np.full((21, 3), np.nan, dtype=np.float32)
            lh_img = np.full((21, 3), np.nan, dtype=np.float32)
            rh_world = np.full((21, 3), np.nan, dtype=np.float32)
            rh_img = np.full((21, 3), np.nan, dtype=np.float32)
            lh_found = False
            rh_found = False

            if hand_res.hand_landmarks and len(hand_res.hand_landmarks) > 0:
                for idx, (hlms, hwlms) in enumerate(zip(hand_res.hand_landmarks, hand_res.hand_world_landmarks)):
                    wrist_x = hlms[0].x
                    wrist_y = hlms[0].y

                    # Calculate distance to Left Wrist (15) and Right Wrist (16)
                    d_left = float("inf")
                    d_right = float("inf")
                    if pose_lw is not None and not np.isnan(pose_lw[0]):
                        d_left = ((wrist_x - pose_lw[0])**2 + (wrist_y - pose_lw[1])**2)**0.5
                    if pose_rw is not None and not np.isnan(pose_rw[0]):
                        d_right = ((wrist_x - pose_rw[0])**2 + (wrist_y - pose_rw[1])**2)**0.5

                    # Fallback to MediaPipe internal handedness if pose wrists unavailable
                    assigned_left = False
                    if d_left < d_right and d_left < 0.25:
                        assigned_left = True
                    elif d_right <= d_left and d_right < 0.25:
                        assigned_left = False
                    elif hand_res.handedness and idx < len(hand_res.handedness):
                        assigned_left = (hand_res.handedness[idx][0].category_name == "Left")

                    arr_w = np.array([[lm.x, lm.y, lm.z] for lm in hwlms], dtype=np.float32)
                    arr_i = np.array([[lm.x, lm.y, lm.z] for lm in hlms], dtype=np.float32)

                    if assigned_left and not lh_found:
                        lh_world = arr_w
                        lh_img = arr_i
                        lh_found = True
                    elif not assigned_left and not rh_found:
                        rh_world = arr_w
                        rh_img = arr_i
                        rh_found = True

            pose_world_list.append(p_world)
            pose_image_list.append(p_img)
            left_hand_world_list.append(lh_world)
            left_hand_image_list.append(lh_img)
            right_hand_world_list.append(rh_world)
            right_hand_image_list.append(rh_img)
            hand_detected_list.append([lh_found, rh_found])

            # Draw overlay if requested
            if out_writer is not None:
                vis_frame = draw_landmarks(frame, p_img, lh_img, rh_img)
                # Overlay status text
                txt = f"F:{frame_idx} | L-Hand:{'OK' if lh_found else 'LOST'} | R-Hand:{'OK' if rh_found else 'LOST'}"
                cv2.putText(vis_frame, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                out_writer.write(vis_frame)

            frame_idx += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()

    # Convert to numpy arrays
    pose_world = np.stack(pose_world_list)         # (N, 33, 4)
    pose_image = np.stack(pose_image_list)         # (N, 33, 4)
    left_hand_world = np.stack(left_hand_world_list) # (N, 21, 3)
    left_hand_image = np.stack(left_hand_image_list) # (N, 21, 3)
    right_hand_world = np.stack(right_hand_world_list) # (N, 21, 3)
    right_hand_image = np.stack(right_hand_image_list) # (N, 21, 3)
    hand_detected = np.array(hand_detected_list, dtype=bool) # (N, 2)

    # Save to out_path (.npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pose_world=pose_world,
        pose_image=pose_image,
        left_hand_world=left_hand_world,
        left_hand_image=left_hand_image,
        right_hand_world=right_hand_world,
        right_hand_image=right_hand_image,
        hand_detected=hand_detected,
        fps=fps,
        frame_count=len(pose_world),
        video_path=str(video_path),
    )

    n_frames = len(pose_world)
    lh_lost = sum(1 for d in hand_detected if not d[0])
    rh_lost = sum(1 for d in hand_detected if not d[1])
    both_lost = sum(1 for d in hand_detected if not d[0] and not d[1])

    stats = {
        "video": str(video_path),
        "out_file": str(out_path),
        "total_frames": n_frames,
        "fps": fps,
        "lh_lost_frames": lh_lost,
        "rh_lost_frames": rh_lost,
        "both_lost_frames": both_lost,
        "lh_lost_pct": round((lh_lost / n_frames) * 100, 1) if n_frames else 0.0,
        "rh_lost_pct": round((rh_lost / n_frames) * 100, 1) if n_frames else 0.0,
        "both_lost_pct": round((both_lost / n_frames) * 100, 1) if n_frames else 0.0,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract 3D pose and hands landmarks using MediaPipe Tasks.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Path to output .npz file")
    parser.add_argument("--overlay", default=None, help="Path to save debug overlay video (.mp4)")
    parser.add_argument("--pose-model", default="data/models/pose_landmarker.task", help="Path to pose_landmarker.task")
    parser.add_argument("--hand-model", default="data/models/hand_landmarker.task", help="Path to hand_landmarker.task")
    args = parser.parse_args()

    v_path = Path(args.video).resolve()
    o_path = Path(args.out).resolve()
    ov_path = Path(args.overlay).resolve() if args.overlay else None

    stats = extract_video_landmarks(
        video_path=v_path,
        out_path=o_path,
        overlay_path=ov_path,
        pose_model_path=Path(args.pose_model).resolve(),
        hand_model_path=Path(args.hand_model).resolve(),
    )

    print(f"Extraction successful: {stats['total_frames']} frames @ {stats['fps']} FPS")
    print(f"Output saved to: {stats['out_file']}")
    print(f"Hands tracking summary:")
    print(f"  - Left Hand lost:  {stats['lh_lost_frames']}/{stats['total_frames']} ({stats['lh_lost_pct']}%)")
    print(f"  - Right Hand lost: {stats['rh_lost_frames']}/{stats['total_frames']} ({stats['rh_lost_pct']}%)")
    print(f"  - Both Hands lost: {stats['both_lost_frames']}/{stats['total_frames']} ({stats['both_lost_pct']}%)")


if __name__ == "__main__":
    main()

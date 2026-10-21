"""
Parametric Dactylology (Fingerspelling) Kinematic Generator
Generates continuous, spline-interpolated 2D/3D MediaPipe landmark sequences
for unmapped words, proper nouns, and manual alphabet tokens (SLSL / English).
"""

import math
from typing import List, Dict, Any, Optional, Tuple


class ParametricFingerspellingGenerator:
    """
    Synthesizes smooth 33-landmark MediaPipe pose sequences for fingerspelled
    tokens using canonical handshape keyframes and cubic S-curve spline easing.
    """

    def __init__(self, fps: float = 10.0, default_hold_frames: int = 3, transition_frames: int = 2):
        self.fps = fps
        self.default_hold_frames = default_hold_frames
        self.transition_frames = transition_frames

        # Standard canonical upright signing template (33 MediaPipe pose landmarks)
        # Landmarked format: [x, y, z, visibility]
        self.base_neutral_pose: List[List[float]] = [
            [0.500, 0.300, -0.80, 0.99],  # 0: nose
            [0.485, 0.280, -0.78, 0.99],  # 1: left_eye_inner
            [0.480, 0.280, -0.78, 0.99],  # 2: left_eye
            [0.475, 0.280, -0.78, 0.99],  # 3: left_eye_outer
            [0.515, 0.280, -0.78, 0.99],  # 4: right_eye_inner
            [0.520, 0.280, -0.78, 0.99],  # 5: right_eye
            [0.525, 0.280, -0.78, 0.99],  # 6: right_eye_outer
            [0.430, 0.300, -0.65, 0.99],  # 7: left_ear
            [0.570, 0.300, -0.65, 0.99],  # 8: right_ear
            [0.480, 0.350, -0.75, 0.99],  # 9: mouth_left
            [0.520, 0.350, -0.75, 0.99],  # 10: mouth_right
            [0.340, 0.500, -0.45, 0.99],  # 11: left_shoulder
            [0.660, 0.500, -0.45, 0.99],  # 12: right_shoulder
            [0.280, 0.720, -0.30, 0.95],  # 13: left_elbow (resting down)
            [0.720, 0.680, -0.35, 0.95],  # 14: right_elbow (bent up toward chest)
            [0.320, 0.880, -0.15, 0.90],  # 15: left_wrist (resting at side)
            [0.620, 0.440, -0.50, 0.98],  # 16: right_wrist (active in signing window)
            [0.320, 0.940, -0.15, 0.90],  # 17: left_pinky
            [0.650, 0.400, -0.49, 0.95],  # 18: right_pinky (default offset)
            [0.330, 0.940, -0.16, 0.90],  # 19: left_index
            [0.620, 0.320, -0.49, 0.95],  # 20: right_index (default offset)
            [0.340, 0.910, -0.15, 0.90],  # 21: left_thumb
            [0.600, 0.380, -0.47, 0.95],  # 22: right_thumb (default offset)
            [0.400, 0.950, 0.00, 0.95],   # 23: left_hip
            [0.600, 0.950, 0.00, 0.95],   # 24: right_hip
            [0.400, 1.350, 0.20, 0.80],   # 25: left_knee
            [0.600, 1.350, 0.20, 0.80],   # 26: right_knee
            [0.400, 1.700, 0.50, 0.70],   # 27: left_ankle
            [0.600, 1.700, 0.50, 0.70],   # 28: right_ankle
            [0.400, 1.750, 0.55, 0.70],   # 29: left_heel
            [0.600, 1.750, 0.55, 0.70],   # 30: right_heel
            [0.400, 1.780, 0.45, 0.70],   # 31: left_foot_index
            [0.600, 1.780, 0.45, 0.70],   # 32: right_foot_index
        ]

        # Keyframe handshape definitions: offsets from active right wrist (L_16)
        # Format: {"index": [dx, dy, dz], "pinky": [dx, dy, dz], "thumb": [dx, dy, dz], "wrist_delta": [dx, dy, dz]}
        self.letter_keyframe_offsets: Dict[str, Dict[str, List[float]]] = {
            # 'A': Fist with thumb upright against side
            "A": {
                "index": [0.010, -0.040, 0.020],
                "pinky": [0.030, -0.030, 0.010],
                "thumb": [-0.025, -0.065, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'B': Flat 4 fingers held straight up, thumb tucked across palm
            "B": {
                "index": [0.000, -0.125, 0.010],
                "pinky": [0.035, -0.105, 0.010],
                "thumb": [-0.015, -0.030, 0.030],
                "wrist_delta": [0.00, -0.01, 0.00],
            },
            # 'C': Curved hand forming a C arc
            "C": {
                "index": [0.020, -0.080, 0.050],
                "pinky": [0.040, -0.060, 0.040],
                "thumb": [-0.030, -0.025, 0.050],
                "wrist_delta": [0.01, 0.00, 0.02],
            },
            # 'D': Index pointing straight up, other fingers forming circle with thumb
            "D": {
                "index": [0.000, -0.130, 0.000],
                "pinky": [0.030, -0.040, 0.020],
                "thumb": [-0.020, -0.050, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'E': Claw/fist with fingertips touching thumb
            "E": {
                "index": [0.015, -0.050, 0.030],
                "pinky": [0.030, -0.040, 0.020],
                "thumb": [-0.010, -0.030, 0.020],
                "wrist_delta": [0.00, 0.01, 0.00],
            },
            # 'F': Index & thumb form circle, pinky and middle extended upward
            "F": {
                "index": [-0.015, -0.055, 0.030],
                "pinky": [0.035, -0.110, 0.010],
                "thumb": [-0.020, -0.055, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'G': Index and thumb pointing horizontally forward/left
            "G": {
                "index": [-0.075, -0.045, 0.030],
                "pinky": [0.020, -0.030, 0.010],
                "thumb": [-0.060, -0.025, 0.030],
                "wrist_delta": [-0.02, 0.00, 0.02],
            },
            # 'H': Index and middle pointing horizontal
            "H": {
                "index": [-0.080, -0.040, 0.020],
                "pinky": [0.020, -0.030, 0.010],
                "thumb": [-0.020, -0.035, 0.020],
                "wrist_delta": [-0.02, 0.00, 0.02],
            },
            # 'I': Pinky straight up, all other fingers closed in a fist
            "I": {
                "index": [0.010, -0.040, 0.020],
                "pinky": [0.030, -0.120, 0.000],
                "thumb": [-0.020, -0.040, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'K': Index vertical, middle forward, thumb wedged between
            "K": {
                "index": [-0.010, -0.120, 0.010],
                "pinky": [0.030, -0.040, 0.020],
                "thumb": [0.000, -0.075, 0.035],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'L': L-shape (index pointing up, thumb pointing horizontal 90 deg)
            "L": {
                "index": [0.000, -0.130, 0.000],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.080, -0.015, 0.020],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'M': Three fingers folded over thumb
            "M": {
                "index": [0.005, -0.035, 0.015],
                "pinky": [0.025, -0.030, 0.010],
                "thumb": [0.010, -0.025, 0.020],
                "wrist_delta": [0.00, 0.01, 0.00],
            },
            # 'N': Two fingers folded over thumb
            "N": {
                "index": [0.005, -0.040, 0.015],
                "pinky": [0.030, -0.030, 0.010],
                "thumb": [-0.005, -0.030, 0.020],
                "wrist_delta": [0.00, 0.01, 0.00],
            },
            # 'O': O-shape (fingertips touching thumb tip)
            "O": {
                "index": [-0.020, -0.060, 0.040],
                "pinky": [0.020, -0.050, 0.030],
                "thumb": [-0.020, -0.050, 0.040],
                "wrist_delta": [0.01, 0.00, 0.02],
            },
            # 'P': Downward inverted K-shape
            "P": {
                "index": [-0.060, 0.040, 0.020],
                "pinky": [0.020, -0.020, 0.010],
                "thumb": [-0.040, 0.020, 0.030],
                "wrist_delta": [-0.01, 0.03, 0.00],
            },
            # 'Q': Downward pointing G-shape
            "Q": {
                "index": [-0.050, 0.050, 0.020],
                "pinky": [0.020, -0.010, 0.010],
                "thumb": [-0.040, 0.040, 0.030],
                "wrist_delta": [-0.01, 0.03, 0.00],
            },
            # 'R': Crossed fingers (index and middle crossed upward)
            "R": {
                "index": [0.005, -0.120, 0.005],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.015, -0.040, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'S': Fist with thumb crossed firmly over middle fingers
            "S": {
                "index": [0.010, -0.040, 0.015],
                "pinky": [0.030, -0.030, 0.010],
                "thumb": [0.005, -0.045, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'T': Fist with thumb wedged between index and middle
            "T": {
                "index": [0.005, -0.045, 0.015],
                "pinky": [0.030, -0.030, 0.010],
                "thumb": [-0.010, -0.050, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'U': Index and middle fingers extended straight up together
            "U": {
                "index": [0.000, -0.125, 0.010],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.020, -0.040, 0.030],
                "wrist_delta": [0.00, -0.01, 0.00],
            },
            # 'V': Index and middle fingers spread in a V / Peace sign
            "V": {
                "index": [-0.025, -0.120, 0.010],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.010, -0.040, 0.030],
                "wrist_delta": [0.00, -0.01, 0.00],
            },
            # 'W': Three fingers spread upward (index, middle, ring)
            "W": {
                "index": [-0.030, -0.115, 0.010],
                "pinky": [0.025, -0.040, 0.020],
                "thumb": [0.000, -0.035, 0.030],
                "wrist_delta": [0.00, -0.01, 0.00],
            },
            # 'X': Index bent into a hook
            "X": {
                "index": [-0.010, -0.080, 0.030],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.020, -0.040, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'Y': Horns / Shaka (thumb and pinky extended, middle 3 curled)
            "Y": {
                "index": [0.010, -0.040, 0.020],
                "pinky": [0.060, -0.090, 0.020],
                "thumb": [-0.070, -0.070, 0.030],
                "wrist_delta": [0.00, 0.00, 0.00],
            },
            # 'Z': Index drawing Z in air
            "Z": {
                "index": [-0.020, -0.110, 0.020],
                "pinky": [0.030, -0.040, 0.010],
                "thumb": [-0.020, -0.040, 0.030],
                "wrist_delta": [0.01, 0.00, 0.01],
            },
        }

        # Sinhala Dactylology Grapheme Mappings
        # In SLSL, Sinhala characters map directly onto related phonetic manual handshapes
        self.sinhala_phonetic_map: Dict[str, str] = {
            "අ": "A", "ආ": "A", "ඇ": "A", "ඈ": "A",
            "ඉ": "I", "ඊ": "I", "උ": "U", "ඌ": "U",
            "එ": "E", "ඒ": "E", "ඔ": "O", "ඕ": "O",
            "ක": "K", "ඛ": "K", "ග": "G", "ඝ": "G",
            "ච": "C", "ඡ": "C", "ජ": "J", "ඣ": "J",
            "ට": "T", "ඨ": "T", "ඩ": "D", "ඪ": "D",
            "ණ": "N", "න": "N",
            "ත": "T", "ථ": "T", "ද": "D", "ධ": "D",
            "ප": "P", "ඵ": "P", "බ": "B", "භ": "B",
            "ම": "M", "ය": "Y", "ර": "R", "ල": "L",
            "ව": "V", "ස": "S", "ශ": "S", "ෂ": "S",
            "හ": "H", "ළ": "L", "ෆ": "F",
        }

    def _get_letter_keyframe(self, char: str) -> List[List[float]]:
        """
        Synthesizes a single 33-landmark pose frame for a specific letter character.
        """
        c = char.upper()
        # Map Sinhala grapheme if present
        if c in self.sinhala_phonetic_map:
            c = self.sinhala_phonetic_map[c]

        offsets = self.letter_keyframe_offsets.get(c, self.letter_keyframe_offsets["B"])

        # Deep copy base neutral template
        pose = [list(pt) for pt in self.base_neutral_pose]

        # Apply wrist delta
        w_dx, w_dy, w_dz = offsets.get("wrist_delta", [0.0, 0.0, 0.0])
        pose[16][0] += w_dx
        pose[16][1] += w_dy
        pose[16][2] += w_dz

        # Adjust right elbow to follow wrist smoothly
        pose[14][0] = (pose[12][0] + pose[16][0]) / 2.0 + 0.05
        pose[14][1] = (pose[12][1] + pose[16][1]) / 2.0 + 0.12

        # Apply handshape offsets relative to adjusted right wrist
        wx, wy, wz = pose[16][0], pose[16][1], pose[16][2]

        idx_off = offsets["index"]
        pose[20] = [round(wx + idx_off[0], 4), round(wy + idx_off[1], 4), round(wz + idx_off[2], 4), 0.95]

        pky_off = offsets["pinky"]
        pose[18] = [round(wx + pky_off[0], 4), round(wy + pky_off[1], 4), round(wz + pky_off[2], 4), 0.95]

        thb_off = offsets["thumb"]
        pose[22] = [round(wx + thb_off[0], 4), round(wy + thb_off[1], 4), round(wz + thb_off[2], 4), 0.95]

        return pose

    def _interpolate_frames(
        self, pose_a: List[List[float]], pose_b: List[List[float]], steps: int
    ) -> List[List[List[float]]]:
        """
        Interpolates between two 33-landmark poses using S-curve (cosine) easing.
        steps: Number of transition frames to generate.
        """
        if steps <= 0:
            return []

        interpolated = []
        for i in range(steps):
            # Normalised progress: tau in (0, 1]
            tau = (i + 1) / (steps + 1)
            # S-curve ease-in ease-out: alpha = (1 - cos(pi * tau)) / 2
            alpha = (1.0 - math.cos(math.pi * tau)) / 2.0

            frame = []
            for j in range(33):
                pt_a = pose_a[j]
                pt_b = pose_b[j]
                x = round((1.0 - alpha) * pt_a[0] + alpha * pt_b[0], 4)
                y = round((1.0 - alpha) * pt_a[1] + alpha * pt_b[1], 4)
                z = round((1.0 - alpha) * pt_a[2] + alpha * pt_b[2], 4)
                vis = round((1.0 - alpha) * pt_a[3] + alpha * pt_b[3], 4)
                frame.append([x, y, z, vis])
            interpolated.append(frame)

        return interpolated

    def generate_letter_frames(
        self, letter: str, hold_frames: Optional[int] = None
    ) -> List[List[List[float]]]:
        """Generates static hold frames for a single fingerspelled letter."""
        hold = hold_frames or self.default_hold_frames
        keyframe = self._get_letter_keyframe(letter)
        return [keyframe for _ in range(hold)]

    def generate_word_frames(
        self,
        word: str,
        hold_frames: Optional[int] = None,
        transition_frames: Optional[int] = None,
    ) -> List[List[List[float]]]:
        """
        Generates a continuous, spline-interpolated motion sequence for an entire
        spelled word (e.g. 'NIPUN' or 'කොළඹ').
        """
        clean_word = word.replace("#", "").strip()
        # Split into individual characters or graphemes
        # If separated by underscore (e.g. '#න_ි_ප_ු_න_්'), split on underscore
        if "_" in clean_word:
            chars = [c for c in clean_word.split("_") if c]
        else:
            chars = [c for c in clean_word if c.strip()]

        if not chars:
            return []

        hold = hold_frames or self.default_hold_frames
        trans = transition_frames or self.transition_frames

        all_frames: List[List[List[float]]] = []

        # Generate entrance transition (from resting hand to first letter)
        first_pose = self._get_letter_keyframe(chars[0])
        entrance_rest = [list(pt) for pt in self.base_neutral_pose]
        # Drop right wrist to resting position initially
        entrance_rest[16] = [0.650, 0.850, -0.15, 0.90]
        entrance_rest[18] = [0.660, 0.910, -0.15, 0.90]
        entrance_rest[20] = [0.650, 0.910, -0.15, 0.90]
        entrance_rest[22] = [0.630, 0.880, -0.15, 0.90]
        all_frames.extend(self._interpolate_frames(entrance_rest, first_pose, steps=trans))

        # Iterate through spelled characters
        current_pose = first_pose
        for idx, char in enumerate(chars):
            target_pose = self._get_letter_keyframe(char)

            # If not first character, interpolate from previous pose
            if idx > 0:
                all_frames.extend(self._interpolate_frames(current_pose, target_pose, steps=trans))

            # Hold the letter pose
            for _ in range(hold):
                all_frames.append(target_pose)

            current_pose = target_pose

        return all_frames

    def generate_token_frames(
        self, gloss: str, original_word: Optional[str] = None
    ) -> List[List[List[float]]]:
        """
        High-level resolver: inspects gloss or original word and synthesizes frames.
        Handles both individual tokens like '#N' and compound spelled tokens like '#N_I_P_U_N'.
        """
        target = gloss
        if gloss.startswith("#"):
            target = gloss[1:]
        elif original_word:
            target = original_word

        return self.generate_word_frames(target)


# Global singleton instance
_generator_instance: Optional[ParametricFingerspellingGenerator] = None


def get_fingerspelling_generator() -> ParametricFingerspellingGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ParametricFingerspellingGenerator()
    return _generator_instance

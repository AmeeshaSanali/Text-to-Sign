import pytest
import numpy as np

try:
    from src.animation.fingerspelling_generator import (
        ParametricFingerspellingGenerator,
        get_fingerspelling_generator,
    )
except ImportError:
    from backend.src.animation.fingerspelling_generator import (
        ParametricFingerspellingGenerator,
        get_fingerspelling_generator,
    )


def test_generator_singleton():
    gen1 = get_fingerspelling_generator()
    gen2 = get_fingerspelling_generator()
    assert gen1 is gen2


def test_generate_single_letter_frame_structure():
    gen = get_fingerspelling_generator()
    frames = gen.generate_letter_frames("A", hold_frames=3)
    assert len(frames) == 3
    # Check 33 landmarks per frame
    first_frame = frames[0]
    assert len(first_frame) == 33
    # Check [x, y, z, vis]
    for lm in first_frame:
        assert len(lm) == 4
        assert 0.0 <= lm[0] <= 1.0  # x coordinate
        assert 0.0 <= lm[1] <= 2.0  # y coordinate
        assert -2.0 <= lm[2] <= 2.0 # z coordinate
        assert 0.0 <= lm[3] <= 1.0  # visibility


def test_dominant_hand_in_signing_window():
    gen = get_fingerspelling_generator()
    frames = gen.generate_letter_frames("B")
    hold_pose = frames[-1]

    # Right wrist (#16)
    r_wrist = hold_pose[16]
    assert 0.50 <= r_wrist[0] <= 0.75  # horizontal chest area
    assert 0.35 <= r_wrist[1] <= 0.60  # chest height

    # Left wrist (#15) resting at side
    l_wrist = hold_pose[15]
    assert l_wrist[1] > 0.80  # lower body height


def test_handshape_distinguishability():
    gen = get_fingerspelling_generator()
    pose_b = gen.generate_letter_frames("B")[-1]  # Flat open hand (index extended up)
    pose_a = gen.generate_letter_frames("A")[-1]  # Closed fist (index curled)

    # In 'B', index tip (#20) is extended significantly higher (lower y) than in 'A'
    index_y_b = pose_b[20][1]
    index_y_a = pose_a[20][1]
    assert index_y_b < index_y_a  # higher in frame = smaller y


def test_generate_word_interpolation():
    gen = get_fingerspelling_generator()
    word_frames = gen.generate_word_frames("NIPUN", hold_frames=3, transition_frames=2)
    # Entrance transition (2) + 5 letters * 3 holds + 4 letter-to-letter transitions * 2 = 25 frames
    assert len(word_frames) == 25
    # Confirm smooth continuity (no sudden landmark teleportation)
    for i in range(1, len(word_frames)):
        prev_wrist = word_frames[i - 1][16]
        curr_wrist = word_frames[i][16]
        delta_x = abs(curr_wrist[0] - prev_wrist[0])
        delta_y = abs(curr_wrist[1] - prev_wrist[1])
        # Maximum per-frame displacement across entire sequence (including entrance)
        assert delta_x < 0.20
        assert delta_y < 0.25

    # Check letter-to-letter stability (after entrance frames >= 2)
    for i in range(3, len(word_frames)):
        prev_wrist = word_frames[i - 1][16]
        curr_wrist = word_frames[i][16]
        # Inter-letter wrist displacement in signing box should be very small
        assert abs(curr_wrist[0] - prev_wrist[0]) < 0.08
        assert abs(curr_wrist[1] - prev_wrist[1]) < 0.08


def test_sinhala_grapheme_mapping():
    gen = get_fingerspelling_generator()
    frames = gen.generate_token_frames("#න_ි_ප_ු_න_්")
    assert len(frames) > 0
    assert len(frames[0]) == 33

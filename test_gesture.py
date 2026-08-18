"""Quick sanity tests for gesture.py -- no camera or mediapipe needed.

Run directly:
    python test_gesture.py
"""

from dataclasses import dataclass

from gesture import classify_gesture


@dataclass
class L:
    x: float
    y: float


def make_landmarks(index_up, middle_up, ring_up, pinky_up):
    """Build a fake 21-point MediaPipe-style landmark list.

    Only the PIP/TIP pairs classify_gesture() actually reads matter;
    everything else is filler at a neutral position.
    """
    lm = [L(0.5, 0.5) for _ in range(21)]
    pairs = {
        "index": (6, 8, index_up),
        "middle": (10, 12, middle_up),
        "ring": (14, 16, ring_up),
        "pinky": (18, 20, pinky_up),
    }
    for pip_idx, tip_idx, is_up in pairs.values():
        if is_up:
            lm[pip_idx] = L(0.5, 0.6)
            lm[tip_idx] = L(0.5, 0.3)  # tip above pip -> extended
        else:
            lm[pip_idx] = L(0.5, 0.3)
            lm[tip_idx] = L(0.5, 0.6)  # tip below pip -> curled
    return lm


def test_fist_is_rock():
    assert classify_gesture(make_landmarks(False, False, False, False)) == "rock"


def test_all_up_is_paper():
    assert classify_gesture(make_landmarks(True, True, True, True)) == "paper"


def test_index_middle_is_scissors():
    assert classify_gesture(make_landmarks(True, True, False, False)) == "scissors"


def test_random_combo_is_unknown():
    assert classify_gesture(make_landmarks(True, False, False, True)) == "unknown"


def test_three_fingers_is_unknown():
    assert classify_gesture(make_landmarks(True, True, True, False)) == "unknown"


if __name__ == "__main__":
    test_fist_is_rock()
    test_all_up_is_paper()
    test_index_middle_is_scissors()
    test_random_combo_is_unknown()
    test_three_fingers_is_unknown()
    print("All gesture tests passed ✅")

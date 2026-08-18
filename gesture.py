"""Gesture classification logic for the AI Rock-Paper-Scissors game.

These functions are independent of MediaPipe/OpenCV so they can be
unit-tested without a webcam, and without even having the mediapipe
package installed. Each "landmark" is any object with .x and .y
attributes (0-1 normalized coordinates) -- the format used by both
MediaPipe's legacy Solutions API and its current Tasks API for each
of the 21 hand points, so this file works unchanged either way.
"""

# MediaPipe Hands landmark indices for the four fingers we use.
# We deliberately ignore the thumb: its motion is mostly horizontal
# rather than vertical, which makes a simple "tip above pip = extended"
# rule unreliable for it. We don't need the thumb to tell rock, paper,
# and scissors apart, so we just skip it.
FINGER_JOINTS = {
    "index": (6, 8),    # (PIP joint index, TIP index)
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}


def finger_states(landmarks) -> dict:
    """Return {finger_name: True/False} for whether each finger is extended.

    A finger counts as "extended" when its fingertip is above (smaller y)
    than its PIP joint -- i.e. the hand is held upright, facing the camera,
    fingers pointing roughly up. That's the pose the game asks players for.
    """
    states = {}
    for name, (pip_idx, tip_idx) in FINGER_JOINTS.items():
        pip_y = landmarks[pip_idx].y
        tip_y = landmarks[tip_idx].y
        states[name] = tip_y < pip_y
    return states


def classify_gesture(landmarks) -> str:
    """Classify a hand pose as 'rock', 'paper', 'scissors', or 'unknown'.

    Rules (based on how many of the 4 non-thumb fingers are extended):
      0 extended             -> rock      (closed fist)
      4 extended              -> paper     (open hand)
      2 extended (index+mid)  -> scissors  (peace-sign shape)
      anything else            -> unknown   (ask the player to try again)
    """
    states = finger_states(landmarks)
    extended = [name for name, is_up in states.items() if is_up]
    count = len(extended)

    if count == 0:
        return "rock"
    if count == 4:
        return "paper"
    if count == 2 and "index" in extended and "middle" in extended:
        return "scissors"
    return "unknown"

"""AI Rock-Paper-Scissors: play against the computer using your webcam.

Uses MediaPipe's legacy "Solutions" API (`mp.solutions.hands`), pinned to
mediapipe==0.10.14 in requirements.txt.

Why not the newer Tasks API? We tried it first (it's the officially
"current" one), but on this project it repeatedly crashed on macOS with a
native (non-Python) error:

    F0000 ... graph_service.h:139] Check failed: service_ Service is unavailable.
    ... DrishtiMetalHelper initWithCalculatorContext ...
    ... TensorsToDetectionsCalculator::Open() ...

That happens because the Tasks API's hand-detection graph tries to
initialize a Metal (GPU) helper for one of its internal calculators, even
when you explicitly request the CPU delegate -- a real bug/edge case in
recent MediaPipe releases (1.0.0/1.0.1 at time of writing) on some Macs.
The older Solutions API uses a different, CPU-only graph under the hood and
never touches Metal/GPU at all, so it sidesteps the crash entirely. It's
also the version almost every hand-tracking tutorial online is built on, so
it's extremely battle-tested. The tradeoff: `mediapipe.solutions` is
deprecated and was removed in mediapipe 0.10.35+/1.0.x, so this project is
intentionally pinned to an older release rather than "latest."

Bonus: this API bundles its own model internally, so unlike the Tasks API
version, there's no separate model file to download on first run.

How it works:
  1. Point your webcam at your hand. A countdown ("3.. 2.. 1.. SHOOT!")
     gives you time to form rock, paper, or scissors.
  2. On "SHOOT!", MediaPipe reads your hand's 21 keypoints, and
     gesture.classify_gesture() turns those points into a move.
  3. The computer picks a random move, a winner is decided, and your
     score updates.

Run:
    python rps_game.py

Controls:
    q          - quit at any time
    any key    - after a round's result is shown, press to start the next round
"""

import random
import time

import cv2
import mediapipe as mp

from gesture import classify_gesture

CHOICES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def decide_winner(player: str, computer: str) -> str:
    if player == computer:
        return "Tie!"
    if BEATS.get(player) == computer:
        return "You win!"
    return "Computer wins!"


def draw_text(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def run_countdown(cap, hands, mp_hands, mp_draw, player_score, computer_score):
    """Show a 3-2-1-SHOOT countdown and return the gesture captured at the end."""
    sequence = ["3", "2", "1", "SHOOT!"]
    start = time.time()
    captured = "unknown"

    while True:
        ok, frame = cap.read()
        if not ok:
            return "unknown"
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        landmarks = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        step = int(time.time() - start)
        if step < len(sequence):
            draw_text(frame, sequence[step], 110, scale=3, color=(0, 255, 255), thickness=4)
        else:
            if landmarks is not None:
                captured = classify_gesture(landmarks)
            draw_text(frame, "You: %d  Computer: %d" % (player_score, computer_score), 450, scale=0.8)
            cv2.imshow("AI Rock Paper Scissors", frame)
            cv2.waitKey(1)
            return captured

        draw_text(frame, "You: %d  Computer: %d" % (player_score, computer_score), 450, scale=0.8)
        cv2.imshow("AI Rock Paper Scissors", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return "quit"


def show_result(cap, result_text, player_score, computer_score):
    """Display the round result for up to 3 seconds, or until a key is pressed."""
    start = time.time()
    while time.time() - start < 3:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        draw_text(frame, result_text, 80, scale=0.85, color=(0, 255, 0))
        draw_text(frame, "You: %d  Computer: %d" % (player_score, computer_score), 450, scale=0.8)
        draw_text(frame, "Press any key for next round, q to quit", 520, scale=0.6, color=(200, 200, 200))
        cv2.imshow("AI Rock Paper Scissors", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key != 255:
            return "continue"
    return "continue"


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions / device index.")

    player_score = 0
    computer_score = 0

    try:
        while True:
            gesture = run_countdown(cap, hands, mp_hands, mp_draw, player_score, computer_score)
            if gesture == "quit":
                break

            computer_choice = random.choice(CHOICES)
            if gesture == "unknown":
                result_text = "Couldn't read your gesture - no points this round."
            else:
                outcome = decide_winner(gesture, computer_choice)
                result_text = "You: %s | Computer: %s -> %s" % (gesture, computer_choice, outcome)
                if outcome == "You win!":
                    player_score += 1
                elif outcome == "Computer wins!":
                    computer_score += 1

            action = show_result(cap, result_text, player_score, computer_score)
            if action == "quit":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()

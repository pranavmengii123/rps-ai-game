# AI Rock-Paper-Scissors (Webcam Edition)

Play rock-paper-scissors against the computer using hand gestures read
live from your webcam. No training data or model-building required —
it uses Google's pre-trained **MediaPipe Hands** model to find 21
keypoints on your hand, then a small rule-based function decides
whether you're showing rock, paper, or scissors.

## A note on which MediaPipe API this uses (read this if you hit errors)

This project uses MediaPipe's **legacy "Solutions" API**
(`mp.solutions.hands`), pinned to `mediapipe==0.10.14` in
`requirements.txt`. That's deliberate, and the opposite of "always use
the newest version" — here's why:

We originally built this on MediaPipe's newer **Tasks API**
(`HandLandmarker`), since that's the one Google now recommends. But on
real testing, it repeatedly crashed on macOS with a native (non-Python)
error, even with the CPU delegate explicitly requested:

```
F0000 ... graph_service.h:139] Check failed: service_ Service is unavailable.
... DrishtiMetalHelper initWithCalculatorContext ...
... TensorsToDetectionsCalculator::Open() ...
```

That's a real bug in recent MediaPipe releases (1.0.0 / 1.0.1 at time of
writing): one of the Tasks API's internal calculators tries to
initialize a Metal (GPU) helper regardless of the delegate you set. The
older Solutions API uses a different, CPU-only graph under the hood and
never touches Metal/GPU at all — and it's the version almost every
hand-tracking tutorial online is built on, so it's extremely
battle-tested. It also bundles its own model internally, so there's no
separate model file to download.

**The tradeoff:** `mediapipe.solutions` was removed starting in
mediapipe 0.10.35 and the 1.0.x line. So don't casually run
`pip install --upgrade mediapipe` in this project's environment — it
will bring back the exact `AttributeError: module 'mediapipe' has no
attribute 'solutions'` error. Stick to the pinned version in
`requirements.txt` unless you're intentionally testing something.

## How it works

1. **MediaPipe Hands** (a pre-trained CNN under the hood) detects your
   hand in each webcam frame and returns 21 (x, y) landmark points —
   fingertips, knuckles, wrist, etc.
2. `gesture.py` looks at 4 of those fingers (index, middle, ring,
   pinky) and checks whether each fingertip is *above* its middle
   knuckle (PIP joint). That tells us if the finger is extended.
3. Based on how many fingers are extended, it classifies the pose:
   - 0 extended → **rock** (fist)
   - 4 extended → **paper** (open hand)
   - index + middle only → **scissors**
   - anything else → **unknown** (try again)
4. `rps_game.py` runs the game loop: a 3-2-1-SHOOT countdown, captures
   your gesture at "SHOOT!", picks a random move for the computer,
   decides the winner, and keeps score.

## Setup

**1. Install Python 3.9–3.12** (a safe, well-supported range for this
pinned MediaPipe version; 3.11 is a solid default if you're not sure).

**2. Create a virtual environment (recommended) and install
dependencies:**

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**3. Sanity-check before running the full game:**

```bash
python -c "import cv2, mediapipe; print(cv2.__version__, mediapipe.__version__)"
```
Should print two version numbers with no error/crash.

**4. Run the game:**

```bash
python rps_game.py
```

A window will open showing your webcam feed with hand landmarks drawn
on top when a hand is detected.

## How to play

- Hold your hand up so the camera can see it clearly, fingers pointing
  up, palm facing the camera.
- Watch the countdown: **3 → 2 → 1 → SHOOT!**
- Show rock, paper, or scissors right as "SHOOT!" appears.
- The result and running score display for 3 seconds, then press any
  key (or just wait) to start the next round.
- Press **q** at any time to quit.

## Files

| File | Purpose |
|---|---|
| `gesture.py` | Pure logic that turns hand landmarks into "rock"/"paper"/"scissors". No camera or mediapipe import needed — easy to unit test. |
| `rps_game.py` | The actual game: webcam capture, MediaPipe Hands detection, countdown UI, scoring. |
| `rps_game.ipynb` | Same game, structured as a Jupyter notebook — one cell per piece, plus a self-test and an emergency-cleanup cell. |
| `test_gesture.py` | Unit tests for the classification logic, using fake landmark data. |
| `requirements.txt` | Dependencies: `mediapipe==0.10.14` (pinned — see note above), `notebook`, `ipykernel`. |

Run the tests any time with:

```bash
python test_gesture.py
```

## Troubleshooting

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
  — you're on a mediapipe version newer than 0.10.14 (someone/something
  ran `pip install --upgrade mediapipe`). Fix: `pip install
  "mediapipe==0.10.14"` again in this environment.
- **Native crash mentioning `DrishtiMetalHelper` / `Service is
  unavailable` / `TensorsToDetectionsCalculator`** — this means
  something is loading the Tasks API instead of this project's code.
  Make sure you're running the current `rps_game.py` (it should import
  `mediapipe as mp` and use `mp.solutions.hands`, not
  `mediapipe.tasks.python`), and delete any `__pycache__` folder.
- **"Could not open webcam"** — another app may be using the camera,
  or you may need to change `cv2.VideoCapture(0)` to `1` in
  `rps_game.py` if you have multiple cameras.
- **Gesture keeps reading "unknown"** — try better lighting, keep your
  whole hand in frame, and make sure fingers are clearly up (open) or
  curled (fist), not half-bent.
- **Segmentation fault on `import cv2, mediapipe`** — you likely have
  both `opencv-python` and `opencv-contrib-python` installed at once
  (mediapipe depends on the latter). Run `pip uninstall opencv-python`
  and reinstall from `requirements.txt` in a fresh venv.
- **Conda's `(base)` environment showing up unexpectedly** — if you
  have Anaconda/Miniconda installed, it may auto-activate on every new
  terminal. Run `conda config --set auto_activate_base false` once to
  stop that, then always `conda deactivate` before activating this
  project's venv.

## Ideas to extend this project

- **Train a real image classifier**: use [Google's Teachable
  Machine](https://teachablemachine.withgoogle.com/) to train your own
  rock/paper/scissors CNN on photos of your own hand, then swap it in
  instead of the landmark-counting rules — a nice next step toward
  "real" ML rather than rule-based logic.
- **Best-of-5 mode**: stop after one player reaches 3 wins and show a
  "Match winner" screen.
- **Sound effects**: play a sound on win/lose/tie using `playsound` or
  `pygame.mixer`.
- **Two-player mode**: use two camera feeds (or one frame split in
  half) so two people can play each other instead of the computer.
- **Difficulty levels**: make the "computer" adapt — e.g., track your
  most common move and counter it, instead of picking randomly.

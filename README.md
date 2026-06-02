# Rock Paper Scissors – Hand Gesture Edition

## Overview

A modern, **glass‑morphic** Rock‑Paper‑Scissors game that uses your webcam and **MediaPipe Hand Landmarker** to recognize hand gestures in real time.  The game features:

- **Vibrant UI** with rounded panels, gradient accents, and animated countdowns.
- **Hand‑gesture detection** (✊ Rock, 🖐 Paper, ✌ Scissors) powered by MediaPipe's TensorFlow‑Lite model.
- **Scoreboard**, round counter, and win/lose/tie feedback.
- **Responsive controls** – Space to play, R to reset, Q/ESC to quit.

## Demo
*(Launch the game to see the UI in action – the webcam window will open.)*

## Prerequisites

- **Python 3.8+** (tested with 3.12)
- A webcam (built‑in laptop cam works fine)
- Recommended: a modern GPU/CPU for smooth hand‑tracking (CPU works as well)

## Installation

1. **Clone the repository** (or copy the `RockPaperScissors` folder into your project).
2. Open a terminal in the folder:
   ```bash
   cd "e:/alhorithm abdul/MeditationApp-master/RockPaperScissors"
   ```
3. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # on Windows
   # source .venv/bin/activate   # on macOS / Linux
   ```
4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Download the MediaPipe hand‑landmarker model** (the script will attempt to download it automatically, but you can do it manually):
   ```bash
   python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'hand_landmarker.task'); print('Model downloaded')"
   ```
   The file `hand_landmarker.task` should appear in the same directory as `game.py`.

## Running the Game

```bash
python game.py
```
A window titled **"Rock Paper Scissors – Hand Gesture"** will appear.

### Controls
| Key | Action |
|-----|--------|
| **Space** | Start a 3‑second countdown and lock in the current hand gesture. |
| **R** | Reset all scores and round counters. |
| **Q / Esc** | Quit the game. |

### Gameplay Flow
1. Show a **Rock**, **Paper**, or **Scissors** hand pose to the webcam. The **Your Gesture** panel (top‑left) updates live.
2. Press **Space** – a 3‑second countdown appears, giving you time to settle on the pose.
3. After the countdown the game picks a random computer move and displays the result (Win / Lose / Tie) with animated graphics.
4. Scores update in the **Scoreboard** (top‑right). Play as many rounds as you like!

## Customisation
- **Colors & Themes** – Adjust the BGR color constants at the top of `game.py` (e.g., `COLOR_ACCENT_CYAN`).
- **Model** – Replace `hand_landmarker.task` with a custom hand‑tracking model (ensure it matches MediaPipe's Tasks API). 
- **Resolution** – Change `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)` and height values for different performance / quality trade‑offs.

## Troubleshooting
- **No webcam feed** – Verify that your camera is not being used by another application. Run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` to test.
- **AttributeError: module 'mediapipe' has no attribute 'solutions'** – This occurs with newer MediaPipe versions. The current code uses the **Tasks API** (`mediapipe.tasks`) which is compatible with v0.10+. Ensure you have the latest version (`pip install -U mediapipe`).
- **Model missing** – If `hand_landmarker.task` is not present, the script will abort with an error message and a download URL. Run the manual download command above.

## License
This project is provided under the **MIT License** – feel free to modify, redistribute, and use it for learning or personal projects.

---
*Enjoy the game and keep rocking those gestures!*

"""
╔══════════════════════════════════════════════════════════════╗
║   ROCK  PAPER  SCISSORS  –  Hand Gesture Edition             ║
║   Uses your webcam + MediaPipe to detect hand gestures       ║
║                                                              ║
║   Controls:                                                  ║
║     SPACE  →  Lock in your gesture & play a round            ║
║     R      →  Reset scores                                   ║
║     Q/ESC  →  Quit                                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import random
import time
import math
import os

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ─── Resolve the model path relative to this script ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# Hand landmark indices (same as the classic MediaPipe 21-point model)
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# Connections for drawing the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm base
]

FINGER_TIPS = {4, 8, 12, 16, 20}

# ─── Color Palette (BGR for OpenCV) ────────────────────────────────────────────
COLOR_BG_DARK        = (30, 25, 20)
COLOR_ACCENT_CYAN    = (230, 200, 0)
COLOR_ACCENT_MAGENTA = (200, 80, 220)
COLOR_ACCENT_GOLD    = (50, 200, 255)
COLOR_WIN            = (100, 220, 80)
COLOR_LOSE           = (80, 80, 230)
COLOR_TIE            = (60, 200, 230)
COLOR_WHITE          = (255, 255, 255)
COLOR_GRAY           = (160, 160, 160)
COLOR_DARK_GRAY      = (100, 100, 100)
COLOR_ROCK           = (100, 140, 230)
COLOR_PAPER          = (220, 180, 80)
COLOR_SCISSORS       = (100, 200, 100)


# ─── Game State ────────────────────────────────────────────────────────────────
class GameState:
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        self.rounds_played = 0
        self.current_gesture = "None"
        self.computer_choice = "None"
        self.result = ""
        self.result_color = COLOR_WHITE
        self.last_play_time = 0
        self.show_result = False
        self.countdown_active = False
        self.countdown_start = 0
        self.countdown_duration = 3  # seconds
        self.locked_gesture = "None"

    def reset(self):
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        self.rounds_played = 0
        self.result = ""
        self.show_result = False
        self.computer_choice = "None"


# ─── Hand Gesture Classifier ──────────────────────────────────────────────────
def classify_gesture(landmarks, handedness_label: str) -> str:
    """
    Classify Rock / Paper / Scissors from a list of 21 NormalizedLandmark.

    Rock     = closed fist  (0-1 fingers extended)
    Paper    = open hand    (4-5 fingers extended)
    Scissors = index + middle extended, ring + pinky closed
    """
    lm = landmarks

    tips      = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pip_joints = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

    fingers_up = []

    # Thumb – compare x (direction depends on hand)
    is_right = (handedness_label == "Right")
    if is_right:
        fingers_up.append(1 if lm[tips[0]].x < lm[pip_joints[0]].x else 0)
    else:
        fingers_up.append(1 if lm[tips[0]].x > lm[pip_joints[0]].x else 0)

    # Index → Pinky – compare y (lower y = higher on screen = extended)
    for i in range(1, 5):
        fingers_up.append(1 if lm[tips[i]].y < lm[pip_joints[i]].y else 0)

    total_up = sum(fingers_up)

    if total_up <= 1:
        return "Rock"
    if total_up >= 4:
        return "Paper"
    if fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 0 and fingers_up[4] == 0:
        return "Scissors"
    if fingers_up[1] == 1 and fingers_up[2] == 1:
        return "Scissors"
    if total_up <= 2:
        return "Rock"
    return "Paper"


# ─── Game Logic ────────────────────────────────────────────────────────────────
def determine_winner(player, computer):
    if player == computer:
        return "IT'S A TIE!", COLOR_TIE
    if (player == "Rock"     and computer == "Scissors") or \
       (player == "Scissors" and computer == "Paper")    or \
       (player == "Paper"    and computer == "Rock"):
        return "YOU WIN!", COLOR_WIN
    return "YOU LOSE!", COLOR_LOSE


# ─── Drawing Utilities ────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=20, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x1 + r, y2 - r), color, thickness)
    cv2.rectangle(overlay, (x2 - r, y1 + r), (x2, y2 - r), color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_gesture_icon(img, gesture, cx, cy, size, color):
    if gesture == "Rock":
        cv2.circle(img, (cx, cy), size, color, -1)
        cv2.circle(img, (cx, cy), size, COLOR_WHITE, 2)
        for dy in (-size // 4, 0, size // 4):
            cv2.line(img, (cx - size // 3, cy + dy), (cx + size // 3, cy + dy), COLOR_WHITE, 2)
    elif gesture == "Paper":
        pts = np.array([[cx - size, cy - size], [cx + size, cy - size],
                        [cx + size, cy + size], [cx - size, cy + size]], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, COLOR_WHITE, 2)
        for i in range(-2, 3):
            lx = cx + i * (size // 3)
            cv2.line(img, (lx, cy - size), (lx, cy - size - size // 2), COLOR_WHITE, 2)
    elif gesture == "Scissors":
        cv2.line(img, (cx - size, cy + size), (cx + size // 2, cy - size), color, 4)
        cv2.line(img, (cx + size, cy + size), (cx - size // 2, cy - size), color, 4)
        cv2.circle(img, (cx, cy + size // 2), size // 3, color, -1)
        cv2.circle(img, (cx, cy + size // 2), size // 3, COLOR_WHITE, 2)


def put_text_centered(img, text, cy, font_scale, color, thickness=2,
                      font=cv2.FONT_HERSHEY_SIMPLEX):
    w = img.shape[1]
    sz = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.putText(img, text, ((w - sz[0]) // 2, cy), font, font_scale,
                color, thickness, cv2.LINE_AA)


def put_text_at(img, text, x, y, font_scale, color, thickness=1,
                font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ─── HUD ──────────────────────────────────────────────────────────────────────
def draw_hud(img, state: GameState, fps: int):
    h, w = img.shape[:2]

    # Top bar
    draw_rounded_rect(img, (10, 10), (w - 10, 65), COLOR_BG_DARK, 15, alpha=0.75)
    put_text_at(img, "ROCK  PAPER  SCISSORS", 25, 47, 0.85, COLOR_ACCENT_CYAN, 2)
    put_text_at(img, f"FPS: {fps}", w - 130, 45, 0.55, COLOR_GRAY)

    # Scoreboard
    pw = 260
    px = w - pw - 15
    draw_rounded_rect(img, (px, 75), (w - 15, 220), COLOR_BG_DARK, 15, alpha=0.7)
    put_text_at(img, "SCOREBOARD", px + 15, 105, 0.6, COLOR_ACCENT_GOLD, 2)
    cv2.line(img, (px + 15, 115), (w - 30, 115), COLOR_DARK_GRAY, 1)
    put_text_at(img, f"You:      {state.player_score}", px + 20, 145, 0.6, COLOR_WIN, 2)
    put_text_at(img, f"Computer: {state.computer_score}", px + 20, 175, 0.6, COLOR_LOSE, 2)
    put_text_at(img, f"Ties:     {state.ties}", px + 20, 205, 0.6, COLOR_TIE, 2)

    # Gesture indicator
    draw_rounded_rect(img, (15, 75), (270, 195), COLOR_BG_DARK, 15, alpha=0.7)
    put_text_at(img, "YOUR GESTURE", 30, 105, 0.55, COLOR_ACCENT_MAGENTA, 2)
    cv2.line(img, (30, 115), (255, 115), COLOR_DARK_GRAY, 1)
    g_color = {"Rock": COLOR_ROCK, "Paper": COLOR_PAPER,
               "Scissors": COLOR_SCISSORS}.get(state.current_gesture, COLOR_GRAY)
    put_text_at(img, state.current_gesture.upper(), 30, 155, 0.9, g_color, 2)
    cv2.circle(img, (240, 148), 18, g_color, -1)
    cv2.circle(img, (240, 148), 18, COLOR_WHITE, 2)

    # Countdown
    if state.countdown_active:
        elapsed = time.time() - state.countdown_start
        remaining = state.countdown_duration - elapsed
        if remaining > 0:
            cnt = int(math.ceil(remaining))
            pulse = 1.0 + 0.3 * math.sin(elapsed * 8)
            draw_rounded_rect(img, (w // 2 - 120, h // 2 - 100),
                              (w // 2 + 120, h // 2 + 60), COLOR_BG_DARK, 25, alpha=0.8)
            put_text_centered(img, str(cnt), h // 2 + 25, 3.0 * pulse, COLOR_ACCENT_GOLD, 4)
            put_text_centered(img, "GET READY!", h // 2 + 55, 0.6, COLOR_GRAY, 1)
        else:
            state.countdown_active = False
            state.locked_gesture = state.current_gesture
            if state.locked_gesture != "None":
                state.computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                state.result, state.result_color = determine_winner(
                    state.locked_gesture, state.computer_choice)
                state.rounds_played += 1
                if "WIN" in state.result:
                    state.player_score += 1
                elif "LOSE" in state.result:
                    state.computer_score += 1
                else:
                    state.ties += 1
                state.show_result = True
                state.last_play_time = time.time()
            else:
                state.result = "NO HAND DETECTED!"
                state.result_color = COLOR_GRAY
                state.show_result = True
                state.last_play_time = time.time()

    # Result
    if state.show_result:
        dt = time.time() - state.last_play_time
        if dt < 4.0:
            alpha = max(0.3, 1.0 - dt / 5.0)
            ry1 = h - 215
            ry2 = h - 15
            draw_rounded_rect(img, (15, ry1), (w - 15, ry2), COLOR_BG_DARK, 20,
                              alpha=min(0.85, alpha))
            put_text_centered(img, state.result, ry1 + 45, 1.1, state.result_color, 3)
            vs_y = ry1 + 90
            you_x = w // 2 - 160
            comp_x = w // 2 + 80
            put_text_at(img, "YOU", you_x, vs_y, 0.5, COLOR_GRAY, 1)
            put_text_at(img, state.locked_gesture.upper(), you_x - 10, vs_y + 35, 0.8, COLOR_WIN, 2)
            draw_gesture_icon(img, state.locked_gesture, you_x + 20, vs_y + 75, 20,
                              {"Rock": COLOR_ROCK, "Paper": COLOR_PAPER,
                               "Scissors": COLOR_SCISSORS}.get(state.locked_gesture, COLOR_GRAY))
            put_text_centered(img, "VS", vs_y + 30, 0.8, COLOR_ACCENT_GOLD, 2)
            put_text_at(img, "CPU", comp_x, vs_y, 0.5, COLOR_GRAY, 1)
            put_text_at(img, state.computer_choice.upper(), comp_x - 10, vs_y + 35, 0.8, COLOR_LOSE, 2)
            draw_gesture_icon(img, state.computer_choice, comp_x + 30, vs_y + 75, 20,
                              {"Rock": COLOR_ROCK, "Paper": COLOR_PAPER,
                               "Scissors": COLOR_SCISSORS}.get(state.computer_choice, COLOR_GRAY))
        else:
            state.show_result = False

    # Controls hint
    if not state.show_result:
        draw_rounded_rect(img, (10, h - 85), (350, h - 10), COLOR_BG_DARK, 12, alpha=0.65)
        put_text_at(img, "[SPACE] Play  [R] Reset  [Q] Quit", 22, h - 30, 0.45, COLOR_GRAY, 1)
        put_text_at(img, "Show Rock, Paper, or Scissors to camera", 22, h - 55, 0.42,
                    COLOR_DARK_GRAY, 1)

    if state.rounds_played > 0:
        put_text_at(img, f"Round {state.rounds_played}", w - 140, h - 25, 0.55, COLOR_DARK_GRAY, 1)


# ─── Styled hand skeleton drawing ─────────────────────────────────────────────
def draw_hand_styled(img, landmarks, gesture):
    color = {"Rock": COLOR_ROCK, "Paper": COLOR_PAPER,
             "Scissors": COLOR_SCISSORS}.get(gesture, COLOR_GRAY)
    h, w = img.shape[:2]

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(pts):
        r = 8 if i in FINGER_TIPS else 4
        cv2.circle(img, (cx, cy), r, color, -1)
        if i in FINGER_TIPS:
            cv2.circle(img, (cx, cy), r, COLOR_WHITE, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN GAME LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  ROCK  PAPER  SCISSORS  —  Hand Gesture Edition")
    print("=" * 60)
    print("  Controls:")
    print("    SPACE  ->  Lock in gesture & play")
    print("    R      ->  Reset scores")
    print("    Q/ESC  ->  Quit")
    print("=" * 60 + "\n")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("  Download it from:")
        print("  https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    # ── Create the HandLandmarker with the new Tasks API ──
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam! Make sure your camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = GameState()
    prev_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Slight darkening for cinematic feel
            frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=10)

            # Convert to MediaPipe Image (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect hand landmarks
            result = landmarker.detect(mp_image)

            state.current_gesture = "None"

            if result.hand_landmarks and result.handedness:
                lm_list = result.hand_landmarks[0]       # first hand
                hand_label = result.handedness[0][0].category_name  # "Left" or "Right"

                state.current_gesture = classify_gesture(lm_list, hand_label)
                draw_hand_styled(frame, lm_list, state.current_gesture)

            # FPS
            now = time.time()
            fps = int(1.0 / max(0.001, now - prev_time))
            prev_time = now

            draw_hud(frame, state, fps)

            cv2.imshow("Rock Paper Scissors - Hand Gesture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and not state.countdown_active:
                state.countdown_active = True
                state.countdown_start = time.time()
                state.show_result = False
            elif key == ord('r'):
                state.reset()

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 40)
    print("         FINAL SCORES")
    print("=" * 40)
    print(f"  You:      {state.player_score}")
    print(f"  Computer: {state.computer_score}")
    print(f"  Ties:     {state.ties}")
    print(f"  Rounds:   {state.rounds_played}")
    print("=" * 40)
    if state.player_score > state.computer_score:
        print("  CONGRATULATIONS! You are the champion!")
    elif state.computer_score > state.player_score:
        print("  The computer wins this time!")
    else:
        print("  It's a perfect tie!")
    print()


if __name__ == "__main__":
    main()

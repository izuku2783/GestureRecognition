import cv2
import time
import subprocess
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 1
GESTURE_HOLD_TIME = 1.0  # seconds
# ---------------------------------------


# -------- Spotify AppleScript helpers --------
def spotify_play_pause():
    subprocess.run(["osascript", "-e",
        'tell application "Spotify" to playpause'])

def spotify_next():
    subprocess.run(["osascript", "-e",
        'tell application "Spotify" to next track'])

def spotify_previous():
    subprocess.run(["osascript", "-e",
        'tell application "Spotify" to previous track'])

def spotify_volume_up():
    subprocess.run(["osascript", "-e",
        'tell application "Spotify" to set sound volume to (sound volume + 10)'])

def spotify_volume_down():
    subprocess.run(["osascript", "-e",
        'tell application "Spotify" to set sound volume to (sound volume - 10)'])


# -------- Gesture → Action --------
def handle_gesture_action(gesture):
    if gesture == "OPEN PALM":
        spotify_play_pause()
    elif gesture == "THUMBS UP":
        spotify_volume_up()
    elif gesture == "THUMBS DOWN":
        spotify_volume_down()
    elif gesture == "PEACE":
        spotify_next()
    elif gesture == "FIST":
        spotify_previous()


# -------- Hand connections --------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# -------- Finger states --------
def get_finger_states(hand):
    tips = [8,12,16,20]
    joints = [6,10,14,18]
    return [1 if hand[t].y < hand[j].y else 0 for t,j in zip(tips,joints)]


# -------- Reliability check --------
def landmarks_are_reliable(hand):
    sep = [abs(hand[t].y - hand[m].y)
           for t,m in zip([8,12,16,20],[5,9,13,17])]
    z_vals = [lm.z for lm in hand]
    return (sum(sep)/4 > 0.04) and (max(z_vals)-min(z_vals) > 0.02)


# -------- Thumb gestures --------
def is_thumbs_up(hand):
    tip, ip, wrist = hand[4], hand[3], hand[0]
    palm_x = np.mean([hand[i].x for i in [5,9,13,17]])
    palm_y = np.mean([hand[i].y for i in [5,9,13,17]])
    return (tip.y < ip.y and tip.y < wrist.y and
            abs(tip.x-palm_x)+abs(tip.y-palm_y) > 0.18)

def is_thumbs_down(hand):
    tip, ip, wrist = hand[4], hand[3], hand[0]
    palm_x = np.mean([hand[i].x for i in [5,9,13,17]])
    palm_y = np.mean([hand[i].y for i in [5,9,13,17]])
    return (tip.y > ip.y and tip.y > wrist.y and
            abs(tip.x-palm_x)+abs(tip.y-palm_y) > 0.18)


# -------- Shape gestures --------
def classify_shape(f):
    if f == [1,1,1,1]: return "OPEN PALM"
    if f == [0,0,0,0]: return "FIST"
    if f == [1,1,0,0]: return "PEACE"
    return "UNKNOWN"


# -------- MediaPipe setup --------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
time.sleep(2)

last_gesture = None
gesture_start = None
triggered = False
start_time = time.time()

# -------- Main loop --------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.time()-start_time)*1000)

    result = landmarker.detect_for_video(mp_image, ts)
    gesture = "UNKNOWN"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        reliable = landmarks_are_reliable(hand)

        if reliable:
            gesture = classify_shape(get_finger_states(hand))
        else:
            if is_thumbs_up(hand):
                gesture = "THUMBS UP"
            elif is_thumbs_down(hand):
                gesture = "THUMBS DOWN"

    now = time.time()
    if gesture != "UNKNOWN":
        if gesture == last_gesture:
            if not triggered and now-gesture_start >= GESTURE_HOLD_TIME:
                handle_gesture_action(gesture)
                triggered = True
        else:
            last_gesture = gesture
            gesture_start = now
            triggered = False
    else:
        last_gesture = None
        triggered = False

    # -------- UI --------
    cv2.putText(frame, f"Gesture: {gesture}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    legend = [
        "OPEN PALM  → Play / Pause",
        "THUMBS UP  → Volume Up",
        "THUMBS DOWN→ Volume Down",
        "PEACE      → Next Track",
        "FIST       → Previous Track"
    ]

    y = 80
    for line in legend:
        cv2.putText(frame, line, (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y += 25

    if gesture == last_gesture and not triggered:
        remain = max(0, GESTURE_HOLD_TIME-(now-gesture_start))
        cv2.putText(frame, f"Hold {gesture}: {remain:.1f}s",
                    (20,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Gesture Spotify Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
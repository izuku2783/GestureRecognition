import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 1   # Change if needed
# ----------------------------------------


# -------- MediaPipe hand connections (official) --------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# -------------------------------------------------------
# 1️⃣ Finger states (index → pinky only)
# -------------------------------------------------------
def get_finger_states(hand):
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    states = []
    for tip, joint in zip(tips, joints):
        states.append(1 if hand[tip].y < hand[joint].y else 0)

    return states  # [index, middle, ring, pinky]


# -------------------------------------------------------
# 2️⃣ Landmark reliability check
# -------------------------------------------------------
def landmarks_are_reliable(hand):
    """
    Conservative check:
    - Fingertips should be sufficiently separated from MCP joints
    - z spread should not collapse (hand not edge-on)
    """

    # Check fingertip separation
    separation = []
    for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        separation.append(abs(hand[tip].y - hand[mcp].y))

    avg_sep = sum(separation) / len(separation)

    # Check z-spread
    z_vals = [lm.z for lm in hand]
    z_spread = max(z_vals) - min(z_vals)

    return avg_sep > 0.04 and abs(z_spread) > 0.02


# -------------------------------------------------------
# 3️⃣ Thumbs-up/down detection (directional, thumb-only)
# -------------------------------------------------------
def is_thumbs_up(hand):
    thumb_tip = hand[4]
    thumb_ip = hand[3]
    wrist = hand[0]

    # Thumb must be extended
    if not (thumb_tip.y < thumb_ip.y):
        return False

    # Thumb must be clearly pointing upward relative to wrist
    if not (thumb_tip.y < wrist.y):
        return False

    # Thumb must be separated from palm
    palm_x = np.mean([hand[i].x for i in [5, 9, 13, 17]])
    palm_y = np.mean([hand[i].y for i in [5, 9, 13, 17]])

    dist = abs(thumb_tip.x - palm_x) + abs(thumb_tip.y - palm_y)
    return dist > 0.18

def is_thumbs_down(hand):
    thumb_tip = hand[4]
    thumb_ip = hand[3]
    wrist = hand[0]

    # Thumb must be extended downward
    if not (thumb_tip.y > thumb_ip.y):
        return False

    if not (thumb_tip.y > wrist.y):
        return False

    palm_x = np.mean([hand[i].x for i in [5, 9, 13, 17]])
    palm_y = np.mean([hand[i].y for i in [5, 9, 13, 17]])

    dist = abs(thumb_tip.x - palm_x) + abs(thumb_tip.y - palm_y)
    return dist > 0.18

# -------------------------------------------------------
# 4️⃣ Shape-based gesture classification
# -------------------------------------------------------
def classify_shape_gesture(fingers):
    if fingers == [1, 1, 1, 1]:
        return "OPEN PALM"
    if fingers == [0, 0, 0, 0]:
        return "FIST"
    if fingers == [1, 1, 0, 0]:
        return "PEACE"
    if fingers == [0, 0, 0, 1]:
        return "THUMBS DOWN"
    return "UNKNOWN"


# -------------------------------------------------------
# 5️⃣ MediaPipe setup
# -------------------------------------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
time.sleep(2)

start_time = time.time()

cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

# -------------------------------------------------------
# 6️⃣ Main loop
# -------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    gesture = "NONE"

    if result.hand_landmarks:
        for hand in result.hand_landmarks:

            # Convert to pixel coords
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            # Draw skeleton
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)

            # Draw & label landmarks
            for i, (x, y) in enumerate(pts):
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            reliable = landmarks_are_reliable(hand)

            if reliable:
                fingers = get_finger_states(hand)
                gesture = classify_shape_gesture(fingers)
            else:
                if is_thumbs_up(hand):
                    gesture = "THUMBS UP"
                elif is_thumbs_down(hand):
                    gesture = "THUMBS DOWN"
                else:
                    gesture = "UNKNOWN"

    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
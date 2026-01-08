import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ---------------- CONFIG ----------------
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 1  # 0 = iPhone, 1 = MacBook camera
# ----------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                               # Palm base
]


# ---------- Finger state (index → pinky only) ----------
def get_finger_states(hand_landmarks):
    fingers = []
    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]

    for tip, joint in zip(finger_tips, finger_joints):
        if hand_landmarks[tip].y < hand_landmarks[joint].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [index, middle, ring, pinky]


# ---------- Hand orientation detection ----------
def get_hand_orientation(hand_landmarks):
    palm_z = (
        hand_landmarks[5].z +
        hand_landmarks[9].z +
        hand_landmarks[13].z +
        hand_landmarks[17].z
    ) / 4

    wrist_z = hand_landmarks[0].z

    return "PALM" if palm_z < wrist_z else "BACK"


# ---------- Thumbs-up (palm-facing) ----------
def is_thumbs_up_palm(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    wrist = hand_landmarks[0]

    palm_x = (
        hand_landmarks[5].x +
        hand_landmarks[9].x +
        hand_landmarks[13].x +
        hand_landmarks[17].x
    ) / 4
    palm_y = (
        hand_landmarks[5].y +
        hand_landmarks[9].y +
        hand_landmarks[13].y +
        hand_landmarks[17].y
    ) / 4

    thumb_up = thumb_tip.y < wrist.y
    thumb_extended = thumb_tip.y < thumb_ip.y

    thumb_distance = abs(thumb_tip.x - palm_x) + abs(thumb_tip.y - palm_y)
    thumb_separated = thumb_distance > 0.18

    closed = 0
    for tip, joint in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if hand_landmarks[tip].y > hand_landmarks[joint].y:
            closed += 1

    return thumb_up and thumb_extended and thumb_separated and closed >= 3


# ---------- Thumbs-up (back-of-hand) ----------
def is_thumbs_up_back(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    wrist = hand_landmarks[0]

    thumb_up = thumb_tip.y > wrist.y
    thumb_extended = thumb_tip.y > thumb_ip.y

    closed = 0
    for tip, joint in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if hand_landmarks[tip].y < hand_landmarks[joint].y:
            closed += 1

    return thumb_up and thumb_extended and closed >= 3


# ---------- MediaPipe setup ----------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)

# ---------- Camera ----------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
time.sleep(2)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

start_time = time.time()
cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

# ---------- Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    gesture = "NONE"
    orientation = ""

    # if result.hand_landmarks:
    #     for hand_landmarks in result.hand_landmarks:
    #         finger_states = get_finger_states(hand_landmarks)
    #         orientation = get_hand_orientation(hand_landmarks)

    #         if orientation == "PALM" and is_thumbs_up_palm(hand_landmarks):
    #             gesture = "THUMBS UP"
    #         elif orientation == "BACK" and is_thumbs_up_back(hand_landmarks):
    #             gesture = "THUMBS UP"
    #         elif finger_states == [0, 0, 0, 0]:
    #             gesture = "FIST"
    #         elif finger_states == [1, 1, 0, 0]:
    #             gesture = "PEACE"
    #         elif finger_states == [1, 1, 1, 1]:
    #             gesture = "PALM"
    #         else:
    #             gesture = "UNKNOWN"

    #         # Draw landmarks
    #         for lm in hand_landmarks:
    #             x = int(lm.x * frame.shape[1])
    #             y = int(lm.y * frame.shape[0])
    #             cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            # Convert normalized landmarks to pixel coords
            points = []
            for i, lm in enumerate(hand_landmarks):
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

                # Draw point
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Label point index
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 255),
                    2
                )


    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Orientation: {orientation}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
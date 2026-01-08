import cv2
import time

# Try different indices
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
# if that fails later, try 2

time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        continue

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

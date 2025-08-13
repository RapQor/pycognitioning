import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Menghitung status jari (1 = terangkat, 0 = turun)
def get_finger_status(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y else 0)

    return fingers

# Menghitung jarak antar landmark
def distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

# Mendeteksi gesture
def detect_gesture(fingers, hand_landmarks, w, h):
    # OK gesture: telunjuk & jempol dekat
    dist_thumb_index = distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8], w, h)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 0, 0, 1]:
        return "Love"
    elif fingers == [0, 1, 0, 0, 1]:
        return "BABYMONSTER"
    elif fingers == [1, 1, 0, 0, 0]:
        return "K-Love"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me Maybe"
    elif fingers == [0, 0, 1, 0, 0]:
        return "FUCK YOU"
    elif dist_thumb_index < 40:  # jarak pixel kecil → membentuk lingkaran
        return "OK"
    else:
        return "Unknown"

# Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ambil status jari
                fingers = get_finger_status(hand_landmarks, hand_label)

                # Deteksi gesture
                gesture = detect_gesture(fingers, hand_landmarks, w, h)

                # Tampilkan hasil
                cv2.putText(frame, f"{hand_label}: {gesture}", (10, 40 if hand_label == "Right" else 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

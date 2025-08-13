import cv2
import mediapipe as mp
import math
import os
import numpy as np

# Path dataset
dataset_path = "dataset"

# Classifier
face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
# LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
label_id = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    label_map[label_id] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detected_faces = face_frontal.detectMultiScale(img, 1.1, 5)
        detected_profiles = face_profile.detectMultiScale(img, 1.1, 5)

        for (x, y, w, h) in detected_faces:
            faces.append(img[y:y+h, x:x+w])
            labels.append(label_id)
        for (x, y, w, h) in detected_profiles:
            faces.append(img[y:y+h, x:x+w])
            labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
print("Training selesai.")

def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # 4 Fingers (Index, Middle, Ring, Pinky)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi semua arah
        detected_faces = []
        detected_faces.extend(face_frontal.detectMultiScale(gray, 1.1, 5))
        detected_faces.extend(face_profile.detectMultiScale(gray, 1.1, 5))

        flipped_gray = cv2.flip(gray, 1)
        right_faces = face_profile.detectMultiScale(flipped_gray, 1.1, 5)
        for (x, y, w, h) in right_faces:
            x_flipped = frame.shape[1] - x - w
            detected_faces.append((x_flipped, y, w, h))

        # Hilangkan kotak duplikat (posisi hampir sama)
        final_faces = []
        for (x, y, w, h) in detected_faces:
            overlap = False
            for (fx, fy, fw, fh) in final_faces:
                if abs(x - fx) < 20 and abs(y - fy) < 20:
                    overlap = True
                    break
            if not overlap:
                final_faces.append((x, y, w, h))

        # Recognition
        for (x, y, w, h) in final_faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)
            name = label_map[label] if confidence <= 70 else "Unknown"
            confidence = confidence if confidence <= 70 else 100 - confidence
            cv2.putText(frame, f"{name} (Kemiripan: {confidence:.0f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Buat bounding box
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

                # Hitung jumlah jari
                finger_count = count_fingers(hand_landmarks, hand_label)

                # Tampilkan jumlah jari & label tangan
                cv2.putText(frame, f"{hand_label} Hand: {finger_count}", 
                            (x_min - 30, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Ambil status jari
                fingers = get_finger_status(hand_landmarks, hand_label)

                # Deteksi gesture
                gesture = detect_gesture(fingers, hand_landmarks, w, h)

                # Tampilkan hasil
                cv2.putText(frame, f"{hand_label}: {gesture}", (10, 40 if hand_label == "Right" else 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition + Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

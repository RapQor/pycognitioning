import cv2
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

# Real-time recognition
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
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

    cv2.imshow("Face Recognition Multi-Angle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

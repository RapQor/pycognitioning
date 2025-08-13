import os
import cv2
import face_recognition
import numpy as np

# --------------------------
# Konfigurasi
# --------------------------
DATASET_DIR = "dataset"         # folder dataset (subfolder per orang)
TOLERANCE = 0.5                 # threshold: lower = stricter (default 0.6 typical)
MODEL = "cnn"                   # "hog" (CPU faster on single core) or "cnn" (slower, more accurate if GPU/dlib compiled with CUDA)
FRAME_RESIZE = 0.25             # scale frame for speed (0.25 = 1/4 size)
PROCESS_EVERY_N_FRAMES = 2      # proses deteksi tiap N frame (untuk performa)

# --------------------------
# Load dataset dan buat encodings
# --------------------------
known_face_encodings = []
known_face_names = []

print("Loading dataset and computing face encodings...")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        # skip non-image files
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png")):
            continue

        image = face_recognition.load_image_file(filepath)
        # bisa ada lebih dari satu wajah pada foto; ambil yang pertama
        encs = face_recognition.face_encodings(image)
        if len(encs) > 0:
            known_face_encodings.append(encs[0])
            known_face_names.append(person_name)
            print(f"  - Loaded {filename} for {person_name}")
        else:
            print(f"  ! No face found in {filepath}, skipping")

print(f"Done. Total known people: {len(set(known_face_names))}, total encodings: {len(known_face_encodings)}")

# --------------------------
# Mulai webcam realtime
# --------------------------
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise RuntimeError("Tidak dapat membuka kamera. Pastikan kamera terpasang dan tidak dipakai aplikasi lain.")

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Keluar.")
        break

    # Resize frame untuk mempercepat pemrosesan
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    # Convert BGR (OpenCV) -> RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = []
    face_encodings = []
    face_names = []

    # Proses hanya tiap N frame untuk efisiensi
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Temukan semua lokasi wajah dan buat encoding
        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # Bandingkan ke encodings known
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            # Gunakan distance untuk memilih match terbaik
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = known_face_names[best_idx]

            face_names.append(name)

    frame_count += 1

    # Gambarkan kotak dan label (kembalikan koordinat ke ukuran asli)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale kembali
        top = int(top / FRAME_RESIZE)
        right = int(right / FRAME_RESIZE)
        bottom = int(bottom / FRAME_RESIZE)
        left = int(left / FRAME_RESIZE)

        # Kotak
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label background
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Teks nama
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1)

    # Info tambahan
    cv2.putText(frame, f"People known: {len(set(known_face_names))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
    cv2.imshow('Real-Time Face Recognition', frame)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

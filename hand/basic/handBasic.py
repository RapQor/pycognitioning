import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,       # Real-time mode
    max_num_hands=2,               # Bisa deteksi 2 tangan
    min_detection_confidence=0.7,  # Akurasi minimal deteksi
    min_tracking_confidence=0.7    # Akurasi minimal tracking
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Ubah dari BGR (OpenCV) ke RGB (MediaPipe)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses deteksi
        results = hands.process(image_rgb)

        # Kalau ada tangan terdeteksi
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar titik & garis tangan
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Tampilkan hasil
        cv2.imshow("Hand Detection", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()

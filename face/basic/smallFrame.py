import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    small_frame = cv2.resize(frame, (640, 480))

    # Tulis jumlah wajah di layar
    cv2.putText(small_frame, f"Wajah terdeteksi: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Latihan 3 - Hitung Wajah', small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

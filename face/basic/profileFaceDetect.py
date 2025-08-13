import cv2

face_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah frontal
    faces_frontal = face_frontal.detectMultiScale(gray, 1.1, 5)

    # Deteksi wajah profil kiri
    faces_profile_left = face_profile.detectMultiScale(gray, 1.1, 5)

    # Deteksi wajah profil kanan (flip image)
    flipped_gray = cv2.flip(gray, 1)
    faces_profile_right = face_profile.detectMultiScale(flipped_gray, 1.1, 5)

    # Gambar hasil frontal
    for (x, y, w, h) in faces_frontal:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Menghadap Depan", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Gambar hasil profil kiri
    for (x, y, w, h) in faces_profile_left:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Menghadap Kiri", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Gambar hasil profil kanan
    for (x, y, w, h) in faces_profile_right:
        x_flipped = frame.shape[1] - x - w
        cv2.rectangle(frame, (x_flipped, y), (x_flipped+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Menghadap Kanan", (x_flipped, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Frontal + Profile Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

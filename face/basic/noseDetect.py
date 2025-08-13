import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_nose.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Tulis jumlah wajah di layar
    cv2.putText(frame, f"Wajah terdeteksi: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 2)

    for (x, y, w, h) in faces:
        # Kotak wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Crop area wajah
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        # ==== Deteksi Mata ====
        eyes = eyes_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(face_roi_color, "Mata", (ex, ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ==== Deteksi Hidung ====
        nose = nose_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(face_roi_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 2)
            cv2.putText(face_roi_color, "Hidung", (nx, ny-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow('Face + Eyes + Nose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

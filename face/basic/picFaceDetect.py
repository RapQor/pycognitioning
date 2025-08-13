import cv2

# 1. Load pre-trained Haar Cascade untuk wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Baca gambar
img = cv2.imread('jungAhyeon.jpeg')  # ganti dengan path fotomu
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ubah jadi grayscale

# 3. Deteksi wajah
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 4. Gambar kotak di wajah
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 5. Tampilkan hasil
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

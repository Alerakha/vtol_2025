import cv2
import time
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8.pt")

# Konfigurasi kamera USB
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Lebar frame
cap.set(4, 480)  # Tinggi frame

while True:
    start_time = time.time()  # Mulai waktu eksekusi

    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dengan YOLOv8
    results = model(frame)

    # Tambahkan FPS ke frame
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil
    results.show()

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

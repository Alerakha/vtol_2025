from ultralytics import YOLO
import cv2

model = YOLO("yolov8.pt")
img = cv2.imread("image.jpg")

# Deteksi objek
results = model(img)

# Tampilkan hasil
results.show()

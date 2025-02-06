from ultralytics import YOLO
import cv2

# Load ONNX model
model = YOLO('/home/krti/yolov8n.onnx')

# Path ke gambar
image_path = 'image.jpg'

# Prediksi
results = model(image_path)

# Load gambar untuk visualisasi
image = cv2.imread(image_path)

# Loop melalui hasil prediksi
for box in results[0].boxes:
    # Ekstrak bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
    conf = box.conf[0]  # Confidence score
    cls = box.cls[0]    # Kelas objek

    # Gambar bounding box di atas gambar
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{results[0].names[int(cls)]} {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan gambar
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

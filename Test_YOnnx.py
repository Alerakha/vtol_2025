import onnxruntime as ort
import numpy as np
import cv2

# Load model ONNX
session = ort.InferenceSession("/home/krti/yolov8n.onnx")

# Preprocessing gambar
img = cv2.imread("image.jpg")
original_height, original_width, _ = img.shape  # Simpan ukuran asli
resized_img = cv2.resize(img, (640, 640))  # Resize ke ukuran model
img_input = resized_img.transpose(2, 0, 1)  # Convert ke format CHW
img_input = img_input.astype(np.float32) / 255.0  # Normalisasi
img_input = np.expand_dims(img_input, axis=0)  # Tambah dimensi batch

# Inference
outputs = session.run(None, {"images": img_input})

# Post-processing untuk menampilkan hasil deteksi
detections = outputs[0][0]  # Ambil hasil deteksi
for detection in detections:
    x_center, y_center, width, height, conf, cls = detection[:6]

    if conf > 0.5:  # Confidence threshold
        # Konversi koordinat dari skala 640x640 ke ukuran gambar asli
        x1 = int((x_center - width / 2) * (original_width / 640))
        y1 = int((y_center - height / 2) * (original_height / 640))
        x2 = int((x_center + width / 2) * (original_width / 640))
        y2 = int((y_center + height / 2) * (original_height / 640))

        # Gambar bounding box
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {int(cls)}: {conf:.2f}"
        cv2.putText(resized_img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan gambar hasil deteksi
cv2.imshow("Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

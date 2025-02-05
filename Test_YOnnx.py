import onnxruntime as ort
import numpy as np
import cv2

# Load model ONNX
session = ort.InferenceSession("yolov8.onnx")

# Preprocessing gambar
img = cv2.imread("image.jpg")
img = cv2.resize(img, (640, 640))  # Sesuaikan ukuran input model
img = img.transpose(2, 0, 1)  # Convert ke format CHW
img = img.astype(np.float32) / 255.0  # Normalisasi
img = np.expand_dims(img, axis=0)

# Inference
outputs = session.run(None, {"images": img})

# Tampilkan hasil deteksi
print(outputs)

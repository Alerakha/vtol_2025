import cv2
import numpy as np
import time
import onnxruntime as ort

# Load model YOLOv8 ONNX
session = ort.InferenceSession("/home/krti/yolov8n.onnx", providers=['CUDAExecutionProvider'])

# Konfigurasi kamera USB
cap = cv2.VideoCapture(0)  # 0 = Kamera USB utama
cap.set(3, 640)  # Lebar frame
cap.set(4, 480)  # Tinggi frame

# Fungsi untuk melakukan preprocessing
def preprocess(img):
    img = cv2.resize(img, (640, 640))  # Sesuaikan ukuran input model
    img = img.transpose(2, 0, 1)  # Convert ke format CHW
    img = img.astype(np.float32) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)
    return img

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_tensor = preprocess(frame)
    # Inference menggunakan ONNX
    outputs = session.run(None, {"images": input_tensor})

    # Dummy hasil deteksi (implementasikan parsing hasil sesuai output model)
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Detected Object (ONNX)", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 20), 2)

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("YOLOv8 ONNX Detection", frame)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

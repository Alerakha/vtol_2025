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

# Kelas objek COCO (80 kelas)
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"]

# Fungsi untuk melakukan preprocessing
def preprocess(img):
    img = cv2.resize(img, (640, 640))  # Sesuaikan ukuran input model
    img = img[:, :, ::-1]  # BGR ke RGB
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW format & normalisasi
    img = np.expand_dims(img, axis=0)  # Tambah dimensi batch
    return img

# Fungsi untuk mendekode hasil YOLOv8
def postprocess(output, frame):
    h, w, _ = frame.shape
    detections = output[0]  # Ambil hasil pertama

    boxes, confidences, class_ids = [], [], []
    
    for i in range(detections.shape[1]):  # Loop semua deteksi
        row = detections[0, i]
        confidence = row[4]  # Confidence score

        if confidence > 0.5:  # Hanya deteksi dengan confidence > 0.5
            x_center, y_center, width, height = row[:4]
            x1 = int((x_center - width / 2) * w / 640)
            y1 = int((y_center - height / 2) * h / 640)
            x2 = int((x_center + width / 2) * w / 640)
            y2 = int((y_center + height / 2) * h / 640)

            class_scores = row[5:]  # Ambil semua skor kelas
            class_id = np.argmax(class_scores)  # Kelas dengan skor tertinggi
            confidence = class_scores[class_id]  # Confidence kelas tersebut

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    return boxes, confidences, class_ids

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_tensor = preprocess(frame)
    
    # Inference menggunakan ONNX
    outputs = session.run(None, {"images": input_tensor})

    # Parsing hasil deteksi
    boxes, confidences, class_ids = postprocess(outputs, frame)

    # Gambar bounding box
    for i in range(min(len(boxes), len(class_ids), len(confidences))):
        x1, y1, x2, y2 = boxes[i]
        label = f"{COCO_CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
        
        # Kotak deteksi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)

    # FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("YOLOv8 ONNX Detection", frame)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

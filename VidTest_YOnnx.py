import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
model_path = "/home/krti/yolov8n.onnx"  # Ganti dengan path model YOLOv8 ONNX Anda
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Load labels (COCO dataset)
model_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
               9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
               16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
               25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
               33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
               49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
               58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
               66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
               74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize & Normalize input
    input_img = cv2.resize(frame, (640, 640))
    input_img = input_img / 255.0  # Normalize
    input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    
    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_img})
    predictions = outputs[0][0]  # YOLOv8 format output
    
    # Process output
    for pred in predictions:
        x, y, w, h, conf, cls = pred[:6]
        if 0 <= int(cls) < len(model_names):
            label = f"{model_names[int(cls)]}: {conf:.2f}"
        else:
            label = f"Unknown Class {int(cls)}: {conf:.2f}"  # Hindari error jika index tidak valid
#            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show output
    frame_resized = cv2.resize(frame, (640,480))
    cv2.imshow("YOLOv8 ONNX Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO

# Load model YOLOv8 yang sudah dilatih
model = YOLO("/home/krti/yolov8n.pt")

# Export ke ONNX
model.export(format="onnx")

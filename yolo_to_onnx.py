from ultralytics import YOLO

# Load model YOLOv8 yang sudah dilatih
model = YOLO("yolov8.pt")

# Export ke ONNX
model.export(format="onnx")

from ultralytics import YOLO

model = YOLO('/home/krti/yolov8n.pt')

result = model('/home/krti/zidane.jpg', save=True)

print(result)

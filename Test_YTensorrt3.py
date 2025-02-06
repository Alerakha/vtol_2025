import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("/home/krti/yolov8n.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Alokasi buffer TensorRT
input_shape = (1, 3, 640, 640)
input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(input_size)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# Fungsi preprocessing untuk gambar
def preprocess(img):
    img = cv2.resize(img, (640, 640))  # Resize ke input model
    img = img.transpose(2, 0, 1)  # Ubah dari HWC ke CHW
    img = img.astype(np.float32) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
    return img

# Fungsi post-processing untuk bounding box
def postprocess(output, img_shape):
    # Placeholder, tergantung output model, perlu disesuaikan dengan decoder YOLOv8
    return []

# Baca gambar
image_path = "image.jpg"
image = cv2.imread(image_path)
input_tensor = preprocess(image)

# Copy ke GPU
cuda.memcpy_htod_async(d_input, input_tensor, stream)

# Waktu inferensi
start_time = time.time()

# Jalankan inferensi dengan TensorRT
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# Copy hasil ke CPU
output_tensor = np.empty(input_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
stream.synchronize()

# Hitung waktu inferensi
fps = 1 / (time.time() - start_time)
print(f"FPS: {fps:.2f}")

# Post-processing untuk mendapatkan bounding box
detections = postprocess(output_tensor, image.shape)

# Gambar bounding box pada gambar
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Class {int(cls)}: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Simpan dan tampilkan hasil
cv2.imwrite("output.jpg", image)
cv2.imshow("YOLOv8 TensorRT", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("/home/krti/yolov8n.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Inisialisasi CUDA memory
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

# Load input image (proses sama seperti ONNX)
img = cv2.imread("image.jpg")
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Copy input ke GPU
cuda.memcpy_htod_async(inputs[0].device, img, stream)

# Inference
context = engine.create_execution_context()
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# Copy hasil dari GPU ke CPU
cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
stream.synchronize()

# Tampilkan hasil
print(outputs)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Logger TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load TensorRT engine
runtime = trt.Runtime(TRT_LOGGER)
with open("/home/krti/yolov8n.trt", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Buat context eksekusi
context = engine.create_execution_context()

# Inisialisasi CUDA memory
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

# Alokasi buffer untuk setiap binding
for binding in engine:
    size = trt.volume(engine.get_tensor_shape(binding)) * np.dtype(np.float32).itemsize
    device_mem = cuda.mem_alloc(size)

    # Simpan buffer berdasarkan jenisnya (input/output)
    if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)
    bindings.append(int(device_mem))

# Load input image
img = cv2.imread("image.jpg")
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1).copy()  # CHW format & contigous
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)  # (1, 3, 640, 640)

# Copy input ke GPU
cuda.memcpy_htod_async(inputs[0], img, stream)

context.set_tensor_address(engine.get_tensor_name(0), inputs[0])

# Inference
context.execute_async_v3(stream_handle=stream.handle)

# Alokasi buffer hasil output di CPU
output_host = np.empty(trt.volume(engine.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)

# Copy hasil dari GPU ke CPU
cuda.memcpy_dtoh_async(output_host, outputs[0], stream)
stream.synchronize()

# Tampilkan hasil
print(output_host)

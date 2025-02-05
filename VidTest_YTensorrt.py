import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("yolov8.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Alokasi buffer untuk input & output TensorRT
input_shape = (1, 3, 640, 640)  # Sesuaikan ukuran input model
input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize

# Alokasi memori CUDA
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(input_size)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# Konfigurasi kamera USB
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Lebar frame
cap.set(4, 480)  # Tinggi frame

# Fungsi preprocessing
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing frame
    input_tensor = preprocess(frame)

    # Copy data ke GPU
    cuda.memcpy_htod_async(d_input, input_tensor, stream)

    # Inference TensorRT
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy hasil kembali ke CPU
    output_tensor = np.empty(input_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
    stream.synchronize()

    # Dummy hasil deteksi (implementasikan parsing hasil sesuai output model)
    cv2.putText(frame, "Detected Object (TensorRT)", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("YOLOv8 TensorRT Detection", frame)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

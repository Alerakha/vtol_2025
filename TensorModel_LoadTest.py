import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("yolov8.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print("Model TensorRT berhasil dimuat!")

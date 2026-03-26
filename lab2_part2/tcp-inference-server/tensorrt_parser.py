import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str) -> trt.ICudaEngine:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [TRT parser error {i}] {parser.get_error(i)}")
            raise RuntimeError(f"TensorRT failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()

    # Set workspace memory (API changed in TRT 8.4)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except AttributeError:
        config.max_workspace_size = 1 << 30  # fallback for older TRT

    # FP16 is faster on Jetson and avoids INT8 calibration
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled.")

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed. Check the warnings above.")
    return engine


class ModelRunner:
    """Wraps a TensorRT engine with its own CUDA stream and pre-allocated
    pinned/device buffers so that inference can be queued asynchronously
    without any per-call memory allocation."""

    def __init__(self, engine: trt.ICudaEngine, input_shape=None):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_shape  = tuple(input_shape) if input_shape is not None else tuple(engine.get_binding_shape(0))
        self.output_shape = tuple(engine.get_binding_shape(1))

        # Page-locked host buffers (faster H<->D transfers)
        self._h_input  = cuda.pagelocked_empty(self.input_shape,  dtype=np.float32)
        self._h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        # Persistent device buffers
        self._d_input  = cuda.mem_alloc(self._h_input.nbytes)
        self._d_output = cuda.mem_alloc(self._h_output.nbytes)

    def infer_async(self, input_data: np.ndarray) -> None:
        """Copy input to device and queue inference — non-blocking."""
        np.copyto(self._h_input, input_data.reshape(self.input_shape).astype(np.float32))
        cuda.memcpy_htod_async(self._d_input, self._h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self._d_input), int(self._d_output)],
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self._h_output, self._d_output, self.stream)

    def synchronize(self) -> np.ndarray:
        """Block until this model's stream is done and return a copy of the output."""
        self.stream.synchronize()
        return self._h_output.copy()


def trt_infer_parallel(
    runner1: "ModelRunner", input1: np.ndarray,
    runner2: "ModelRunner", input2: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Run two models sequentially on their own CUDA streams.

    On the Jetson Nano (single Maxwell SM) TensorRT execution contexts
    share internal scratch/workspace memory.  Queuing both before
    synchronizing either lets their kernels overlap and corrupt that
    shared workspace, leading to 'unspecified launch failure'.
    Running them back-to-back avoids the conflict with negligible
    overhead since the hardware serializes kernels anyway.

    Returns:
        (output1, output2) matching runner1 and runner2 respectively.
    """
    runner1.infer_async(input1)    # queue on stream-1
    out1 = runner1.synchronize()   # drain stream-1 before touching the GPU again

    runner2.infer_async(input2)    # queue on stream-2
    out2 = runner2.synchronize()   # drain stream-2

    return out1, out2


if __name__ == "__main__":
    import time

    engine1 = build_engine("models/yolov12n.onnx")
    engine2 = build_engine("models/steering_model.onnx")

    runner1 = ModelRunner(engine1)
    runner2 = ModelRunner(engine2)

    print(f"Model 1 - input: {runner1.input_shape}  output: {runner1.output_shape}")
    print(f"Model 2 - input: {runner2.input_shape}  output: {runner2.output_shape}")

    dummy1 = np.random.rand(*runner1.input_shape).astype(np.float32)
    dummy2 = np.random.rand(*runner2.input_shape).astype(np.float32)

    t0 = time.time()
    out1, out2 = trt_infer_parallel(runner1, dummy1, runner2, dummy2)
    print(f"Parallel inference time: {time.time() - t0:.3f}s")
    print("Output 1 shape:", out1.shape)
    print("Output 2 shape:", out2.shape)
import numpy as np
import struct
import pickle
import socket
import time

from tensorrt_parser import build_engine, ModelRunner, trt_infer_parallel


class SocketServer:
    ## Following code is inspired from https://github.com/ekbanasolutions/numpy-using-socket
    ## to transmit python objects (numpy array) through a TCP socket with addition to be
    ## resilient to connection lost (meaning other end process has been killed)

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", 42000))
        self._sock.listen(1)
        self._conn = None

        self._receive_data = b''
        self._payload_size = struct.calcsize("!Q")

    def connected(self):
        return self._conn is not None

    def accept_client(self):
        print("Waiting new client")
        self._conn, addr = self._sock.accept()
        self._receive_data = b''  # clear stale bytes from any previous connection
        print("Client connected")

    def send_result(self, result):
        data = pickle.dumps(result)
        msg_size = struct.pack("!Q", len(data))

        try:
            self._conn.sendall(msg_size + data)
        except (ConnectionResetError, BrokenPipeError):
            self._connection_lost()

    def receive_image(self):
        try:
            # Retrieve the message size data
            while len(self._receive_data) < self._payload_size:
                data = self._conn.recv(4096)
                if data:
                    self._receive_data += data
                else:
                    self._connection_lost()
                    return None

            # Convert the data to get the message size
            packed_msg_size = self._receive_data[:self._payload_size]
            msg_size = struct.unpack("!Q", packed_msg_size)[0] + self._payload_size

            # Retrieve the full message
            while len(self._receive_data) < msg_size:
                data = self._conn.recv(4096)
                if data:
                    self._receive_data += data
                else:
                    self._connection_lost()
                    return None

            # Extract the data and prepare for next
            data = self._receive_data[self._payload_size:msg_size]
            self._receive_data = self._receive_data[msg_size:]

            # Convert the message to numpy
            return pickle.loads(data)

        except (ConnectionResetError, BrokenPipeError):
            self._connection_lost()
            return None

    def _connection_lost(self):
        print("Lost connection with client")
        self._conn.close()
        self._conn = None


def main():
    print("Hello from TCP inference server!")

    YOLO_MODEL_PATH = "models/yolov11_wins_fp32.onnx"
    STEERING_MODEL_PATH = YOLO_MODEL_PATH#"models/steering_model.onnx"
    yolo_runner = None
    steering_runner = None

    print("Building TensorRT engines...")
    yolo_engine = build_engine(YOLO_MODEL_PATH)
    steering_engine = build_engine(STEERING_MODEL_PATH)

    print("Starting server...")
    serv = SocketServer()

    print("Ready to accept client connections")
    while True:
        if not serv.connected():
            serv.accept_client()

        if serv.connected():
            data = serv.receive_image()
            if data is None:
                continue

            image = data["image"]

            # Initialize each model's runner on first use.
            # Each runner owns its CUDA stream + pre-allocated pinned/device buffers.
            # The input shape comes from the ONNX engine binding (set during export)
            if yolo_runner is None:
                print("Preparing TensorRT ModelRunner for YOLO...")
                yolo_runner = ModelRunner(yolo_engine)
                print(f"YOLO - input: {yolo_runner.input_shape}  output: {yolo_runner.output_shape}")
            if steering_runner is None:
                print("Preparing TensorRT ModelRunner for steering model...")
                steering_runner = ModelRunner(steering_engine)
                print(f"Steering - input: {steering_runner.input_shape}  output: {steering_runner.output_shape}")

            # The client sends a preprocessed float32 NCHW tensor (1,3,640,640),
            # so we just ensure it is contiguous and in float32.
            model_input = np.ascontiguousarray(image.astype(np.float32))

            # Both models are queued on independent CUDA streams before either
            # is synchronized, allowing the GPU to run them in parallel
            start_time = time.time()
            yolo_output, steering_output = trt_infer_parallel(
                yolo_runner, model_input,
                steering_runner, model_input,
            )
            print(f"Parallel TensorRT inference time: {time.time() - start_time:.3f} seconds")

            # Send both results together as a single dict
            serv.send_result({"detection": yolo_output, "steering": steering_output})


if __name__ == "__main__":
    main()

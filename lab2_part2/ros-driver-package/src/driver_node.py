#!/usr/bin/env python3

"""
driver_node.py — Camera Proxy ROS Node

This node acts as a bridge between the Duckiebot's camera and an external
AI inference server (e.g. a YOLO object-detection model).

How it fits into the ROS ecosystem:
  - ROS (Robot Operating System) is a middleware framework that lets separate
    processes (called "nodes") communicate by publishing and subscribing to
    named "topics". Think of a topic as a typed message bus.
  - This node *subscribes* to the camera topic so it receives a new image every
    time the camera driver publishes one.
  - It then forwards that image to an external server over a plain TCP socket,
    waits for the inference result, and processes it.

Entry point: the `if __name__ == '__main__'` block at the bottom of this file.
"""

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
from cv_bridge import CvBridge

# Standard library modules used for the TCP socket communication with the
# inference server.
import socket
import pickle  # serialises Python objects (e.g. numpy arrays) into bytes
import struct  # packs/unpacks fixed-size binary headers (used for message framing)

# Address and port of the external inference server. 127.0.0.1 means the
# server is expected to run on the same machine (localhost).
SERV_ADDR = "127.0.0.1"
SERV_PORT = 42000

# Class names corresponding to the YOLO model's class IDs (0-6).
YOLO_CLASS_NAMES = {
    0: 'Duckie',
    1: 'Duckiebot',
    2: 'Traffic light',
    3: 'QR code',
    4: 'Stop sign',
    5: 'Intersection sign',
    6: 'Signal sign',
}


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------
# A ROS node is a single process that performs a specific task. Nodes
# communicate with each other through topics (pub/sub) or services (req/res).
#
# CameraProxyNode inherits from DTROS (which itself inherits from rospy's
# infrastructure). By calling super().__init__() we register this process as
# a node with the ROS master, after which it can publish, subscribe, etc.
class CameraProxyNode(DTROS):

    def __init__(self, node_name):
        # Register this process as a ROS node. The node_name is the identifier
        # visible in tools like `rosnode list` or `rqt_graph`.
        # NodeType.GENERIC means it does not fit a more specific category
        # (e.g. PERCEPTION, CONTROL, …).
        super(CameraProxyNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Read the robot's name from the environment. On a Duckiebot, this is
        # set automatically (e.g. "mybot"). It is used to build the topic name.
        self._vehicle_name = os.environ['VEHICLE_NAME']

        # In ROS, a *topic* is a named channel over which nodes exchange
        # messages. The camera driver publishes compressed images on this topic.
        # The naming convention is: /<vehicle_name>/<node_name>/<stream>.
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # CvBridge provides helpers to convert ROS image messages ↔ cv2 arrays.
        self._bridge = CvBridge()

        # rospy.Subscriber tells ROS: "whenever a message arrives on
        # self._camera_topic, call self.callback with the message as argument".
        # The callback runs in a background thread managed by rospy; our main
        # loop (run()) runs concurrently in the main thread.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # --- TCP socket state ---
        self._socket = None          # the active socket object
        self._connected = False      # whether we currently have a connection
        self._waiting_result = False # True while we wait for inference output
        self._receive_data = b''     # accumulator for incoming bytes
        # Number of bytes in the fixed-size length header that precedes every
        # message. We use "!Q" (network byte-order unsigned 64-bit int) so
        # the header is always 8 bytes regardless of the platform (32-bit ARM
        # vs 64-bit x86/aarch64).
        self._payload_size = struct.calcsize("!Q")

        # Latest camera frame as a numpy array. Updated by the subscriber
        # callback; read by the main loop when sending to the server.
        self._image = None
        self._msg = None

        # Letterbox preprocessing metadata — set by _preprocess_for_yolo(),
        # used by _scale_boxes_to_original() to map detections back to the
        # original image coordinate space.
        self._letterbox_gain = 1.0       # scale ratio (model / original)
        self._letterbox_pad = (0, 0)     # (pad_x, pad_y) in pixels

        self.pub_detections = rospy.Publisher(
            "~detections/image/compressed",
            CompressedImage,
            queue_size=1
        )


    def run(self):
        """
        Main loop of the node.

        In ROS, nodes are typically structured around a loop that runs at a
        controlled frequency. This method implements that loop:
          1. Make sure we have a TCP connection to the inference server.
          2. If we sent a frame and are waiting for the result, try to receive it.
          3. If we are idle (no pending result), send the latest camera frame.

        The loop rate (10 Hz) is decoupled from the camera's publication rate;
        the subscriber callback updates self._image in the background whenever
        a new frame arrives.
        """
        # rospy.Rate controls how fast the loop runs. Rate(10) → 10 iterations
        # per second (one iteration every 100 ms). rate.sleep() at the end of
        # each iteration sleeps exactly long enough to maintain this frequency.
        rate = rospy.Rate(10)

        # rospy.is_shutdown() returns True once a shutdown signal has been
        # received (e.g. Ctrl-C or `rosnode kill`). This is the idiomatic way
        # to write a ROS main loop that exits cleanly.
        while not rospy.is_shutdown():
            # Ensure we are connected to server
            if (not self._connected):
                self._connect()

            try:
                # If we previously sent a frame, try to read the inference
                # result from the server. The socket is non-blocking so this
                # returns quickly even if data has not arrived yet.
                if (self._connected and self._waiting_result):
                        result = self._receive_result()

                        if result is not None:
                            self._process_result(result)

                # Send the most recent camera frame to the server and mark
                # that we are now waiting for a result.
                if (self._connected and not self._waiting_result):
                    self._send_image()

            except (ConnectionResetError, BrokenPipeError):
                # The server closed the connection unexpectedly; reset state so
                # we attempt to reconnect on the next loop iteration.
                self._connection_lost()

            except BlockingIOError:
                # The socket is non-blocking. When no data is available yet,
                # recv() raises BlockingIOError instead of blocking the thread.
                # We simply skip this iteration and retry next cycle.
                pass

            # Sleep until the next iteration is due, keeping the loop at 10 Hz.
            rate.sleep()


    def callback(self, msg):
        """
        ROS subscriber callback — called automatically by rospy every time a
        new CompressedImage message is published on the camera topic.

        This function runs in a separate thread (managed by rospy's internal
        thread pool), so it executes concurrently with run(). Writing to a
        single Python reference (self._image) is safe due to the GIL, but be
        aware of this concurrency if you add more complex logic here.

        Args:
            msg (CompressedImage): ROS message containing the JPEG-compressed
                                   camera frame and its metadata.
        """
        # Decode the compressed ROS message into an OpenCV-compatible numpy
        # array (BGR uint8, shape H×W×3). This is the format expected by cv2
        # and by the inference server.
        self._image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self._msg = msg


    def _process_yolo_outputs(self, outputs, conf_threshold=0.5, iou_threshold=0.4):
        """Process raw YOLO model outputs (shape: (1, 11, 6300))

        The 11 rows are: [cx, cy, w, h, cls0, cls1, cls2, cls3, cls4, cls5, cls6].
        In YOLOv11 there is no separate objectness score; the class scores are
        the final confidences.

        Mirrors the ultralytics postprocessing pipeline:
          1. Transpose → (6300, 11)
          2. Confidence filter
          3. Convert cx,cy,w,h → x1,y1,x2,y2 (xyxy) for NMS
          4. cv2 NMS (uses x,y,w,h format internally)
          5. Return detections in xyxy format: [x1, y1, x2, y2, conf, class_id]

        Note: returned boxes are in *model input space* (480×640). Call
        _scale_boxes_to_original() to map them to the original image.

        Args:
            outputs: Raw model output array with shape (1, 11, 6300).
            conf_threshold (float): Minimum confidence to keep a detection.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression.

        Returns:
            list: Filtered detections, each as [x1, y1, x2, y2, conf, class_id]
                  in model input coordinates.
        """
        # Squeeze batch dim and transpose: (1, 11, 6300) → (6300, 11)
        preds = outputs[0].T

        # Split bounding-box coordinates and class scores
        boxes_cxcywh = preds[:, :4]   # (6300, 4) — cx, cy, w, h
        class_scores = preds[:, 4:]   # (6300, 7)

        # Best class confidence and id per anchor
        confidences = np.max(class_scores, axis=1)   # (6300,)
        class_ids = np.argmax(class_scores, axis=1)  # (6300,)

        # Keep only anchors above the confidence threshold
        mask = confidences > conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        # Convert (cx, cy, w, h) → (x1, y1, w, h) for cv2.dnn.NMSBoxes
        boxes_xywh = boxes_cxcywh.copy()
        boxes_xywh[:, 0] -= boxes_cxcywh[:, 2] / 2  # x1 = cx - w/2
        boxes_xywh[:, 1] -= boxes_cxcywh[:, 3] / 2  # y1 = cy - h/2

        # Non-Maximum Suppression (cv2 expects [x, y, w, h] = top-left + size)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes_xywh[i]
                # Store as xyxy (matching ultralytics output format)
                detections.append([
                    float(x1), float(y1), float(x1 + w), float(y1 + h),
                    float(confidences[i]), int(class_ids[i]),
                ])

        if detections:
            detected_classes = [YOLO_CLASS_NAMES.get(int(d[5]), 'Unknown') for d in detections]
            rospy.loginfo(f"Detected {len(detections)} object(s): {', '.join(detected_classes)}")
        else:
            rospy.loginfo("No objects detected")

        return detections


    def _scale_boxes_to_original(self, detections, orig_shape):
        """Rescale detection boxes from model-input space to original image space.

        Mirrors ultralytics ops.scale_boxes(img1_shape, boxes, img0_shape):
          1. Subtract the letterbox padding offsets
          2. Divide by the scale gain
          3. Clip to the original image boundaries

        Uses self._letterbox_gain and self._letterbox_pad stored by
        _preprocess_for_yolo() during preprocessing.

        Args:
            detections (list): Detections as [x1, y1, x2, y2, conf, cls_id]
                               in model input coordinates.
            orig_shape (tuple): Original image shape (H, W) or (H, W, C).

        Returns:
            list: Detections rescaled to original image coordinates.
        """
        if not detections:
            return detections

        gain = self._letterbox_gain
        pad_x, pad_y = self._letterbox_pad
        orig_h, orig_w = orig_shape[:2]

        scaled = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det

            # Remove padding offset and undo scaling
            x1 = (x1 - pad_x) / gain
            y1 = (y1 - pad_y) / gain
            x2 = (x2 - pad_x) / gain
            y2 = (y2 - pad_y) / gain

            # Clip to original image boundaries (matches clip_boxes)
            x1 = max(0.0, min(x1, orig_w))
            y1 = max(0.0, min(y1, orig_h))
            x2 = max(0.0, min(x2, orig_w))
            y2 = max(0.0, min(y2, orig_h))

            scaled.append([x1, y1, x2, y2, conf, cls_id])

        return scaled

        
    def _draw_detections(self, image, detections):
        """Draw detection boxes and labels on image
        
        Args:
            image (np.ndarray): Original image
            detections (list): List of detections in xyxy format
                               [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            
            # Draw rectangle (xyxy format)
            color = (0, 0, 255)
            
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )
            
            # Add label
            label = f"{YOLO_CLASS_NAMES.get(int(class_id), 'Unknown')} {conf:.2f}"
            cv2.putText(
                annotated_image,
                label,
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        # Add status text if object detected
        cv2.putText(
            annotated_image,
            "Object detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
            
        return annotated_image


    def _publish_detections(self, image, header):
        """Publish annotated image
        
        Args:
            image (np.ndarray): Annotated image
            header: Original message header
        """
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        self.pub_detections.publish(msg)
    
    
    def _process_result(self, result):
        """
        Handle the inference result returned by the external server.

        Args:
            result (dict): Deserialized server response. Expected to contain a
                           "detection" key holding raw YOLO output objects.
        """
        # rospy.loginfo() is the ROS equivalent of print(). Messages are
        # forwarded to the /rosout topic and to the terminal, making them
        # visible in tools like `rqt_console` or `rostopic echo /rosout`.
        rospy.loginfo(f"camera_proxy: Got new result with shapes {result['steering'].shape} (Steering) and {result['detection'].shape} (Detection)")

        # Parse the raw YOLO outputs into a clean list of detections
        # (boxes in model-input space: 480×640).
        detections = self._process_yolo_outputs(result["detection"])

        # Rescale detection boxes from model-input space (480×640) back to
        # the original camera image coordinates — mirrors ultralytics
        # ops.scale_boxes(img.shape[2:], pred[:,:4], orig_img.shape).
        detections = self._scale_boxes_to_original(detections, self._image.shape)

        # Draw detections on image
        # annotated_image = self._draw_detections(self._image, detections)
        
        # Publish annotated image
        # self._publish_detections(annotated_image, self._msg.header)

        # TODO: do something with the result



    def _preprocess_for_yolo(self, image, target_h=480, target_w=640):
        """Preprocess a BGR uint8 image for YOLO inference.

        Replicates the ultralytics preprocessing pipeline exactly
        (LetterBox with auto=False, center=True + predictor.preprocess)
        so the ONNX / TensorRT model receives identical input:
          1. Letterbox resize (maintain aspect ratio, center-pad with 114)
          2. BGR → RGB
          3. HWC (H, W, 3) → NCHW (1, 3, H, W)
          4. Normalize pixel values to [0, 1] float32

        Also stores letterbox metadata (gain, pad offsets) as instance
        attributes for use by _scale_boxes_to_original() during
        postprocessing.

        Args:
            image (np.ndarray): BGR uint8 image of shape (H, W, 3).
            target_h (int): Model input height  (default 480).
            target_w (int): Model input width   (default 640).

        Returns:
            np.ndarray: Float32 tensor of shape (1, 3, target_h, target_w).
        """
        h, w = image.shape[:2]

        # Scale ratio (new / old) — matches ultralytics LetterBox
        r = min(target_h / h, target_w / w)

        # Resized (unpadded) dims — ultralytics uses round(), NOT int()
        new_w = round(w * r)
        new_h = round(h * r)

        # Padding needed on each axis
        dw = target_w - new_w
        dh = target_h - new_h

        # Center the image (split padding to both sides) — center=True default
        dw /= 2.0
        dh /= 2.0

        # Resize only if dimensions changed
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        # Exact padding amounts — matches ultralytics rounding convention
        top = round(dh - 0.1)
        bottom = round(dh + 0.1)
        left = round(dw - 0.1)
        right = round(dw + 0.1)

        # Apply border padding (gray 114, same as ultralytics)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Store letterbox parameters for postprocessing (scale_boxes)
        self._letterbox_gain = r
        self._letterbox_pad = (left, top)  # (pad_x, pad_y)

        # BGR → RGB via numpy slice (same as ultralytics: im[..., ::-1])
        rgb = padded[..., ::-1]

        # HWC → CHW, add batch dim → (1, 3, H, W), normalise to [0, 1]
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        tensor /= 255.0

        return np.ascontiguousarray(tensor)


    ## Folowing code is inspired from https://github.com/ekbanasolutions/numpy-using-socket
    ## to transmit python objects (numpy array) through a TCP socket with addition to be
    ## resilient to connection lost (meaning other end process has been killed)

    def _send_image(self):
        """
        Serialise the latest camera frame and send it to the inference server.

        The message format is a simple length-prefixed protocol:
          [ 8-byte unsigned long (payload length) ][ pickle payload ]
        This lets the receiver know exactly how many bytes to read for the
        full message, avoiding ambiguity in a streaming TCP connection.
        """
        # Guard: the callback may not have fired yet (e.g. camera not started).
        if (self._image is None):
            rospy.loginfo("camera_proxy: No image to send to model")
            return

        # Take a local reference to avoid a race condition with the
        # subscriber callback, which updates self._image from another thread.
        image = self._image

        # Preprocess the raw BGR camera frame into the format the ONNX/TensorRT
        # YOLO model expects: RGB, letterbox-resized to 640×640, normalised
        # [0,1] float32, NCHW layout → shape (1, 3, 640, 640).
        preprocessed = self._preprocess_for_yolo(image)

        # Serialise the preprocessed tensor to bytes using pickle.
        data = pickle.dumps({"image": preprocessed, "shape": preprocessed.shape, "dtype": str(preprocessed.dtype)})

        # Prepend a fixed-size header containing the length of the payload.
        msg_size = struct.pack("!Q", len(data))

        # sendall() must complete atomically — switch to blocking mode so the
        # full ~5 MB payload is guaranteed to be sent before we return.
        # (Non-blocking sendall can raise BlockingIOError mid-transfer when the
        # kernel send buffer fills up, causing the server to receive a partial
        # message and corrupting the framing.)
        self._socket.setblocking(True)
        self._socket.sendall(msg_size + data)
        self._socket.setblocking(False)

        # Record that we are now waiting for the server's response before
        # sending the next frame (simple request/response flow).
        self._waiting_result = True


    def _receive_result(self):
        """
        Read the inference result from the server using the same
        length-prefixed protocol used by _send_image().

        Because the socket is non-blocking, recv() may return less data than
        requested (or raise BlockingIOError if nothing is available). We
        accumulate bytes in self._receive_data across multiple calls until the
        full message has arrived.

        Returns:
            dict | None: Deserialised result dict, or None if the connection
                         was lost while reading.
        """
        # Phase 1: accumulate at least enough bytes to read the header.
        # self._payload_size is the fixed size of the length prefix (8 bytes).
        while len(self._receive_data) < self._payload_size:
            data = self._socket.recv(4096)  # read up to 4096 bytes at a time
            if data:
                self._receive_data += data
            else:
                # recv() returning empty bytes means the server closed the
                # connection gracefully.
                self._connection_lost()
                return None

        # Phase 2: decode the header to learn the total message length.
        # msg_size is the total number of bytes we need to have buffered
        # (header + payload) before we can deserialise.
        packed_msg_size = self._receive_data[:self._payload_size]
        msg_size = struct.unpack("!Q", packed_msg_size)[0] + self._payload_size

        # Phase 3: keep reading until we have the full payload.
        while len(self._receive_data) < msg_size:
            data = self._socket.recv(4096)
            if data:
                self._receive_data += data
            else:
                self._connection_lost()
                return None

        # Phase 4: slice out exactly one message and advance the buffer so
        # any extra bytes belong to the next message (pipelining safety).
        data = self._receive_data[self._payload_size:msg_size]
        self._receive_data = self._receive_data[msg_size:]
        self._waiting_result = False

        # Deserialise the bytes back into a Python object (dict with results).
        return pickle.loads(data)


    def _connect(self):
        """
        Attempt to open a TCP connection to the inference server.

        The connection attempt itself uses a short timeout (1 s) so we do not
        stall the ROS loop for too long if the server is not yet ready. Once
        connected, the socket is switched to non-blocking mode so that recv()
        in _receive_result() returns immediately instead of blocking the loop.
        """
        try:
            # Create a standard IPv4 TCP socket.
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Allow up to 1 second for the connection handshake. If the server
            # is not listening, connect() will raise socket.timeout.
            self._socket.settimeout(1)
            self._socket.connect((SERV_ADDR, SERV_PORT))

            rospy.loginfo("camera_proxy: Connected to server")

            # Switch to non-blocking mode: recv() will now raise
            # BlockingIOError immediately if no data is available, instead of
            # sleeping until data arrives (which would freeze the ROS loop).
            self._socket.setblocking(False)
            self._connected = True
            self._waiting_result = False
            self._receive_data = b''  # clear any stale data from a previous session
        except (ConnectionRefusedError, ConnectionResetError, socket.timeout):
            # Server is not reachable; clean up and try again on the next loop
            # iteration (run() will call _connect() again).
            self._socket.close()


    def _connection_lost(self):
        """
        Handle an unexpected loss of the TCP connection.

        Resets connection state so the next iteration of run() will attempt to
        reconnect via _connect().
        """
        rospy.loginfo("camera_proxy: Lost connection with server")
        self._socket.close()
        self._connected = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# ROS nodes are launched either directly (python driver_node.py) or through a
# launch file. In both cases, Python sets __name__ == '__main__' for the
# top-level script, so this block is the canonical entry point.
if __name__ == '__main__':

    # Instantiate the node. The constructor registers it with the ROS master
    # and sets up the subscriber, but does NOT start the main loop yet.
    node = CameraProxyNode(node_name='camera_proxy_node')

    # Start our custom main loop (10 Hz send/receive cycle).
    node.run()

    # rospy.spin() blocks here and keeps the Python process alive so that
    # rospy's background threads (handling subscriber callbacks, etc.) can keep
    # running. It returns only when a shutdown signal is received.
    # NOTE: because run() contains `while not rospy.is_shutdown()`, it will
    # exit before spin() gets a chance to do much — spin() is here as a
    # safety net in case run() returns early for any reason.
    rospy.spin()

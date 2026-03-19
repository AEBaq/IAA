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
        # message (an unsigned long packed by struct).
        self._payload_size = struct.calcsize("L")

        # Latest camera frame as a numpy array. Updated by the subscriber
        # callback; read by the main loop when sending to the server.
        self._image = None


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


    def _process_yolo_outputs(self, image, outputs):
        """Process YOLO model outputs
        
        Args:
            image (np.ndarray): Original image
            outputs (list): Raw model outputs
            
        Returns:
            list: Processed detections in format [x, y, w, h, conf, class_id]
        """
        detections = []
        
        # Process each prediction
        for out in outputs:
            # Get boxes, confidences and class IDs
            boxes = out.boxes
            
            # Convert boxes to desired format
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Add detection
                detections.append([x1, y1, w, h, conf, cls_id])

                rospy.loginfo(f"Detection: Class={self.class_names[cls_id]}, Confidence={conf:.2f}, Box=({x1:.0f}, {y1:.0f}, {w:.0f}, {h:.0f})")
                
        return detections
    
    
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
        rospy.loginfo(f"camera_proxy: Got new result with shape {type(result)}")

        # Parse the raw YOLO outputs into a clean list of detections.
        detections = self._process_yolo_outputs(self._image, result["detection"])

        # TODO: do something with the result, for now we just log it
        # Possible next steps: publish detections on a new ROS topic, overlay
        # bounding boxes on the image, trigger a motor command, etc.



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

        # Serialise the image dict to bytes using pickle. We include shape and
        # dtype so the server can reconstruct the numpy array correctly.
        data = pickle.dumps({"image": self._image, "shape": self._image.shape, "dtype": str(self._image.dtype)})

        # Prepend a fixed-size header containing the length of the payload.
        # struct.pack("L", n) encodes n as an unsigned long (8 bytes on 64-bit).
        msg_size = struct.pack("L", len(data))

        # sendall() ensures all bytes are written even if the OS only accepts
        # part of the buffer in a single write call.
        self._socket.sendall(msg_size + data)

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
        msg_size = struct.unpack("L", packed_msg_size)[0] + self._payload_size

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

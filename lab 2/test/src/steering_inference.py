#!/usr/bin/env python3
"""
Steering inference node for Duckiebot.
Controls the robot using a trained SteeringNet model.
"""

import os
import rospkg
import rospy
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from cv_bridge import CvBridge

# YOUR_CODE_HERE: import model


def predict(model, image, preprocess, device="cpu"):
    """
    Run inference on an image.
    image: PIL Image or numpy array (H, W, 3) RGB
    Returns: vel_left, vel_right (float32)
    """
    if hasattr(image, "convert"):
        img = image.convert("RGB")
    else:
        img = PILImage.fromarray(image).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    vel_left, vel_right = out[0].cpu().numpy()
    return vel_left, vel_right


class SteeringInferenceNode:
    def __init__(self):
        rospy.init_node("steering_inference")

        # Parameters
        pkg_path = rospkg.RosPack().get_path("test")
        default_checkpoint = os.path.join(pkg_path, "src/best_model.pth")
        self.checkpoint_path = rospy.get_param(
            "~checkpoint_path",
            default_checkpoint,
        )
        self.robot_name = os.environ['VEHICLE_NAME']
        self.throttle_factor = rospy.get_param("~throttle_factor", 1)  # process every Nth frame
        self.clamp_min = rospy.get_param("~clamp_min", 0.0)
        self.clamp_max = rospy.get_param("~clamp_max", 1.0)

        if not os.path.isfile(self.checkpoint_path):
            rospy.logerr(f"Checkpoint not found: {self.checkpoint_path}")
            raise FileNotFoundError(self.checkpoint_path)

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # YOUR_CODE_HERE: import model
        #self.model = ...
        print(f"Loading model from {self.checkpoint_path}")
        
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )

        self.model.load_state_dict(ckpt["model_state_dict"])
        rospy.loginfo("Loaded model weights")
        self.model.to(self.device)
        self.model.eval()

        # YOUR_CODE_HERE: Preprocessing (must match training)
        #self.preprocess = ...

        self.bridge = CvBridge()
        self.frame_count = 0

        # Topics
        camera_topic = f"/{self.robot_name}/camera_node/image/compressed" 
        wheels_topic = f"/{self.robot_name}/wheels_driver_node/wheels_cmd"

        self.sub = rospy.Subscriber(
            camera_topic, CompressedImage, self.compressed_callback, queue_size=1
        )

        self.pub = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1
        )

        rospy.loginfo(
            f"Steering inference started. Subscribed to {camera_topic}, "
            f"publishing to {wheels_topic}"
        )
        rospy.spin()

    def _process_and_publish(self, pil_image):
        vel_left, vel_right = predict(
            self.model, pil_image, self.preprocess, self.device
        )
        print(f"vel_left: {vel_left}, vel_right: {vel_right}")
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.vel_left = float(vel_left)
        cmd.vel_right = float(vel_right)
        self.pub.publish(cmd)

    def compressed_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.throttle_factor != 0:
            return
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            rospy.logwarn_throttle(5.0, "Failed to decode compressed image")
            return
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image)
        self._process_and_publish(pil_image)

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.throttle_factor != 0:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        pil_image = PILImage.fromarray(cv_image)
        self._process_and_publish(pil_image)


if __name__ == "__main__":
    try:
        SteeringInferenceNode()
    except rospy.ROSInterruptException:
        pass

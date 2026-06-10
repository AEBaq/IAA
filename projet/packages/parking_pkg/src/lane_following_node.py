#!/usr/bin/env python3
"""
Lane following node for DuckieBot - Lab 4 (P4 Smart Parking Reverse)
Basé sur steering_interference.py du Lab 2.
Utilise le modèle ResNet18 fine-tuné pour prédire vel_left et vel_right.
"""

import os
import rospy
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped


# ===========================================================================
# Architecture du modèle (identique au Lab 2)
# ===========================================================================
class SteeringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def init_and_load(checkpoint_path: str) -> "SteeringNet":
        model = SteeringNet()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


# ===========================================================================
# Nœud ROS
# ===========================================================================
class LaneFollowingNode:
    def __init__(self):
        self.node_name = rospy.get_name()
        self.robot_name = rospy.get_param("~robot_name", "d1")

        # Traiter 1 frame sur N (évite la surcharge CPU du Jetson)
        self.throttle_factor = rospy.get_param("~throttle_factor", 2)
        self.frame_count = 0

        # Flag activé/désactivé par la state machine
        self.active = True

        # ---------------------------------------------------------------
        # Chargement du modèle
        # ---------------------------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"[{self.node_name}] Using device: {self.device}")

        # Chemin vers le modèle (copié dans models/ du repo)
        default_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../../models/best_finetuned_model.pt",
        )
        model_path = rospy.get_param("~model_path", default_model_path)

        if not os.path.isfile(model_path):
            rospy.logerr(f"[{self.node_name}] Modèle introuvable : {model_path}")
            raise FileNotFoundError(model_path)

        self.model = SteeringNet.init_and_load(model_path)
        self.model.to(self.device)
        rospy.loginfo(f"[{self.node_name}] Modèle chargé depuis {model_path}")

        # ---------------------------------------------------------------
        # Preprocessing (identique à l'entraînement Lab 2)
        # ---------------------------------------------------------------
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # ---------------------------------------------------------------
        # ROS pub/sub
        # ---------------------------------------------------------------
        self.sub_image = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size=2**24,
        )

        self.pub_wheels = rospy.Publisher(
            f"/{self.robot_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1,
        )

        rospy.loginfo(f"[{self.node_name}] Lane following node prêt.")

    # ------------------------------------------------------------------
    # Callback caméra
    # ------------------------------------------------------------------
    def cb_image(self, msg):
        if not self.active:
            return

        self.frame_count += 1
        if self.frame_count % self.throttle_factor != 0:
            return

        # Décodage
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            rospy.logwarn_throttle(5.0, f"[{self.node_name}] Échec décodage image")
            return

        # BGR → RGB → PIL
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image)

        # Inférence + publication
        vel_left, vel_right = self.predict(pil_image)
        self.publish_wheels(vel_left, vel_right)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def predict(self, pil_image):
        x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        vel_left, vel_right = out[0].cpu().numpy()
        return float(vel_left), float(vel_right)

    # ------------------------------------------------------------------
    # Publication roues
    # ------------------------------------------------------------------
    def publish_wheels(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.pub_wheels.publish(msg)

    def stop(self):
        """Arrêt complet — appelé par la state machine ou on_shutdown."""
        self.publish_wheels(0.0, 0.0)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    rospy.init_node("lane_following_node", anonymous=False)
    node = LaneFollowingNode()
    rospy.on_shutdown(node.stop)
    rospy.spin()

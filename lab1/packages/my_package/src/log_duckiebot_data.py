#!/usr/bin/env python3

import os
import csv
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np

class DuckiebotLogger:
    def __init__(self):
        rospy.init_node("duckiebot_data_logger")

        # Output dirs
        self.output_dir = rospy.get_param("~output_dir", "dataset")
        print(self.output_dir)
        #check if output_dir is a valid directory, if not create it
        if not os.path.isdir(self.output_dir):
            print(f"Output directory {self.output_dir} does not exist, creating it")
            os.makedirs(self.output_dir, exist_ok=True)

        self.image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        #print current directory
        self.csv_path = os.path.join(self.output_dir, "labels.csv")
        self.robot_name = os.environ['VEHICLE_NAME']
        # CSV init
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestamp", "image_name", "vel_left", "vel_right"]
        )

        # Subscribers
        image_sub = Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
        )
        wheels_sub = Subscriber(
            f"/{self.robot_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped
        )

        # Time sync
        ats = ApproximateTimeSynchronizer(
            [image_sub, wheels_sub],
            queue_size=10,
            slop=0.1  # 50 ms tolerance
        )
        ats.registerCallback(self.callback)

        rospy.loginfo("Duckiebot data logger started.")
        rospy.spin()

    def callback(self, img_msg, wheels_msg):
        # Timestamp
        #print('AAAAAAAAH',img_msg.header.stamp)
        stamp = img_msg.header.stamp.to_sec()
        image_name = f"{stamp:.6f}.jpg"
        image_path = os.path.join(self.image_dir, image_name)

        # Decode compressed image
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            rospy.logwarn("Failed to decode image")
            return

        # Save image
        cv2.imwrite(image_path, image)

        # Save label
        self.csv_writer.writerow([
            stamp,
            image_name,
            wheels_msg.vel_left,
            wheels_msg.vel_right
        ])
        self.csv_file.flush()

if __name__ == "__main__":
    try:
        DuckiebotLogger()
    except rospy.ROSInterruptException:
        pass


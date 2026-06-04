#!/usr/bin/env python3
"""
Nœud principal - P4 Smart Parking (Reverse)
Orchestre le lane following, la détection AprilTag, et la manœuvre de parking.
"""

import rospy
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, AprilTagDetectionArray
from state_machine import StateMachine, State


class ParkingNode:
    def __init__(self):
        rospy.init_node("parking_node", anonymous=False)
        self.node_name = rospy.get_name()

        # ---------------------------------------------------------------
        # Paramètres
        # ---------------------------------------------------------------
        self.robot_name = rospy.get_param("~robot_name", "d1")
        self.target_tag_id = rospy.get_param("~target_tag_id", 0)  # ID AprilTag du spot
        self.base_speed = rospy.get_param("~base_speed", 0.3)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.25)
        self.wait_duration = rospy.get_param("~wait_duration", 5.0)  # secondes

        # Durées des phases odométriques (à calibrer sur le vrai robot)
        self.align_duration = rospy.get_param("~align_duration", 1.0)  # s
        self.reverse_duration = rospy.get_param("~reverse_duration", 2.5)  # s
        self.exit_duration = rospy.get_param("~exit_duration", 2.5)  # s

        rospy.loginfo(f"[{self.node_name}] Target AprilTag ID : {self.target_tag_id}")

        # ---------------------------------------------------------------
        # State machine
        # ---------------------------------------------------------------
        self.sm = StateMachine()

        # ---------------------------------------------------------------
        # Import du nœud de lane following
        # ---------------------------------------------------------------
        from lane_following_node import LaneFollowingNode

        self.lane_following = LaneFollowingNode()

        # ---------------------------------------------------------------
        # ROS pub/sub
        # ---------------------------------------------------------------
        self.pub_wheels = rospy.Publisher(
            f"/{self.robot_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1,
        )

        self.sub_tags = rospy.Subscriber(
            f"/{self.robot_name}/apriltag_detector_node/detections",
            AprilTagDetectionArray,
            self.cb_apriltag,
            queue_size=1,
        )

        # Timers pour les phases odométriques
        self._phase_timer = None

        # Boucle principale à 10 Hz
        self.rate = rospy.Rate(10)
        rospy.on_shutdown(self.stop)
        rospy.loginfo(f"[{self.node_name}] Parking node prêt.")

    # ==================================================================
    # Callback AprilTag
    # ==================================================================
    def cb_apriltag(self, msg):
        """Détecte si le tag cible est visible."""
        if not self.sm.is_lane_following():
            return

        for detection in msg.detections:
            if detection.tag_id == self.target_tag_id:
                rospy.loginfo(
                    f"[{self.node_name}] AprilTag {self.target_tag_id} détecté !"
                )
                self.sm.on_tag_detected()
                return

    # ==================================================================
    # Boucle principale
    # ==================================================================
    def run(self):
        while not rospy.is_shutdown():
            state = self.sm.state

            if state == State.LANE_FOLLOWING:
                # Le lane following tourne en autonomie via son propre callback
                self.lane_following.active = True

            elif state == State.PARKING_DETECTED:
                # On stoppe le lane following et on commence l'alignement
                self.lane_following.active = False
                self.stop()
                rospy.sleep(0.5)
                self.sm.on_start_align()
                self._start_phase(
                    duration=self.align_duration,
                    vel_left=self.base_speed * 0.5,
                    vel_right=-self.base_speed * 0.5,  # légère rotation pour s'aligner
                    on_done=self.sm.on_aligned,
                )

            elif state == State.ALIGNING:
                # Phase gérée par le timer → on attend
                pass

            elif state == State.REVERSE_PARKING:
                # Phase gérée par le timer → on attend
                pass

            elif state == State.WAITING:
                # Phase gérée par le timer → on attend
                pass

            elif state == State.EXIT:
                # Phase gérée par le timer → on attend
                pass

            self.rate.sleep()

    # ==================================================================
    # Gestion des phases odométriques (open-loop)
    # ==================================================================
    def _start_phase(self, duration, vel_left, vel_right, on_done):
        """
        Lance une commande roues pendant `duration` secondes,
        puis appelle on_done() et enchaîne la prochaine phase.
        """
        # Annule un timer précédent si existant
        if self._phase_timer is not None:
            self._phase_timer.shutdown()

        self.publish_wheels(vel_left, vel_right)

        def _cb(event):
            self.stop()
            on_done()
            self._on_phase_done()

        self._phase_timer = rospy.Timer(rospy.Duration(duration), _cb, oneshot=True)

    def _on_phase_done(self):
        """Enchaîne automatiquement la bonne phase suivante."""
        state = self.sm.state

        if state == State.REVERSE_PARKING:
            rospy.loginfo(
                f"[{self.node_name}] Recul terminé — attente {self.wait_duration}s"
            )
            self._start_phase(
                duration=self.wait_duration,
                vel_left=0.0,
                vel_right=0.0,
                on_done=self.sm.on_wait_done,
            )

        elif state == State.WAITING:
            rospy.loginfo(f"[{self.node_name}] Attente terminée — sortie du parking")
            self._start_phase(
                duration=self.exit_duration,
                vel_left=self.base_speed,
                vel_right=self.base_speed,
                on_done=self.sm.on_exit_done,
            )

        elif state == State.EXIT:
            rospy.loginfo(
                f"[{self.node_name}] Sortie terminée — reprise lane following"
            )
            self.lane_following.active = True

    def _start_reverse(self):
        """Lance la phase de recul (appelée après l'alignement)."""
        rospy.loginfo(f"[{self.node_name}] Début recul...")
        self._start_phase(
            duration=self.reverse_duration,
            vel_left=self.reverse_speed,
            vel_right=self.reverse_speed,
            on_done=self.sm.on_parked,
        )

    # ==================================================================
    # Publication roues
    # ==================================================================
    def publish_wheels(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.pub_wheels.publish(msg)

    def stop(self):
        self.publish_wheels(0.0, 0.0)


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    node = ParkingNode()

    # Quand l'alignement est détecté, lancer le recul
    original_on_aligned = node.sm.on_aligned

    def on_aligned_with_reverse():
        original_on_aligned()
        node._start_reverse()

    node.sm.on_aligned = on_aligned_with_reverse

    node.run()

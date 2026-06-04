#!/usr/bin/env python3
"""
State machine pour P4 - Smart Parking (Reverse)
États : LANE_FOLLOWING → PARKING_DETECTED → ALIGNING → REVERSE_PARKING → WAITING → EXIT → LANE_FOLLOWING
"""

import rospy
from enum import Enum


class State(Enum):
    LANE_FOLLOWING = "LANE_FOLLOWING"
    PARKING_DETECTED = "PARKING_DETECTED"
    ALIGNING = "ALIGNING"
    REVERSE_PARKING = "REVERSE_PARKING"
    WAITING = "WAITING"
    EXIT = "EXIT"


class StateMachine:
    def __init__(self):
        self.state = State.LANE_FOLLOWING
        rospy.loginfo(f"[StateMachine] État initial : {self.state.value}")

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def transition(self, new_state: State):
        rospy.loginfo(f"[StateMachine] {self.state.value} → {new_state.value}")
        self.state = new_state

    # ------------------------------------------------------------------
    # Méthodes de test d'état (lisibilité dans parking_node.py)
    # ------------------------------------------------------------------
    def is_lane_following(self):
        return self.state == State.LANE_FOLLOWING

    def is_parking_detected(self):
        return self.state == State.PARKING_DETECTED

    def is_aligning(self):
        return self.state == State.ALIGNING

    def is_reverse_parking(self):
        return self.state == State.REVERSE_PARKING

    def is_waiting(self):
        return self.state == State.WAITING

    def is_exit(self):
        return self.state == State.EXIT

    # ------------------------------------------------------------------
    # Transitions nommées (appelées depuis parking_node.py)
    # ------------------------------------------------------------------
    def on_tag_detected(self):
        """AprilTag du bon spot détecté pendant le lane following."""
        if self.is_lane_following():
            self.transition(State.PARKING_DETECTED)

    def on_start_align(self):
        """Début de la phase d'alignement avant recul."""
        if self.is_parking_detected():
            self.transition(State.ALIGNING)

    def on_aligned(self):
        """Alignement terminé, on commence à reculer."""
        if self.is_aligning():
            self.transition(State.REVERSE_PARKING)

    def on_parked(self):
        """Manœuvre de recul terminée, on attend 5s."""
        if self.is_reverse_parking():
            self.transition(State.WAITING)

    def on_wait_done(self):
        """Attente de 5s terminée, on sort du parking."""
        if self.is_waiting():
            self.transition(State.EXIT)

    def on_exit_done(self):
        """Sortie du parking terminée, reprise du lane following."""
        if self.is_exit():
            self.transition(State.LANE_FOLLOWING)

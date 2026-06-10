#!/bin/bash
source /environment.sh
dt-launchfile-init
# ----------------------------------------------------------------------------
export ROS_MASTER_URI=http://d1.local:11311
export ROS_IP=$(hostname -I | awk '{print $1}')

dt-exec roslaunch parking_pkg parking.launch \
    robot_name:=d1 \
    target_tag_id:=0
# ----------------------------------------------------------------------------
dt-launchfile-join
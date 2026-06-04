#!/bin/bash
source /environment.sh
# initialize launch file
dt-launchfile-init
# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# Changer target_tag_id selon l'ID de ton AprilTag de parking
dt-exec roslaunch parking_pkg parking.launch \
    robot_name:=${VEHICLE_NAME} \
    target_tag_id:=0

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE
# wait for app to end
dt-launchfile-join

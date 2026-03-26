To collect dataset on duckiebot matrix (if real robot just skip the robot creation then follow from step 4 ):
1-Create a virtual duckiebot : dts duckiebot virtual create --type duckiebot --configuration DB21J robotname
2-Run matrix: dts matrix run --standalone --embedded --map sandbox
3-dts matrix attach myduckiebot map_0/vehicle_0 
4-dts duckiebot demo --demo_name lane_following --duckiebot_name robotname --package_name duckietown_demos
5-dts duckiebot keyboard_control robotname, et activer le autopilot
6-go dans le folder qui contient log_duckiebot_data.py, rajouter dans le dockerfile: RUN mkdir -p "/data" 
7-dts devel build -f
8-dts devel run -R myduckiebot -- -v /path/on_lcoal_laptop:/data (make sure that your launcher contains rosrun my_package log_duckiebot_data.py _output_dir:=/data)
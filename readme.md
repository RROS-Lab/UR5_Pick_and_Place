# Pick and Plae Pipeline using  UR5 with Robotiq gripper and Realsense Camera

1. Launch UR5 driver.
In ur_common.launch, modify by deleting 

<!-- Convert joint states to /tf tranforms -->
<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>,
  
since we want to use our own URDF that includes the robotiq and realsense.

roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.100   kinematics_confqig:=$(rospack find ur_calibration)/etc/my_robot_calibration.yaml

(Go to UR5 screen) - follow directions there
a) Go to init screen
b) Power ON - Start - OK
b) Run Program
c) File - Load Program
d) mmnewer.urp
e) Play button (Terminal should read connected to reverse interface)
  
  
2. launch the gripper driver for arduino

roslaunch robotiq_2f_gripper_control gripper_bringup.launch

  
3. Moveit configuration launch

roslaunch ur5_with_robotiq_moveit_config ur5_gripper_moveit_planning_execution.launch 


RViz 
a) Fixed Frame - base_link
b) Add - Motion planning



4. RGBD camera realsense node with pointcloud

    roslaunch realsense2_camera rs_camera.launch filters:=pointcloud

5. Launch perception node for manipulation
    roslaunch simple_grasping basic_perception_dbg.launch 

6. Run python file
  rosrun ur5_python_moveit_wrapper move_group_python_interface_tutorial.py
  


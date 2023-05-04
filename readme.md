# Pick and Plae Pipeline using  UR5 with Robotiq gripper and Realsense Camera

## 1. Launch UR5 driver.
In ur_common.launch, modify by deleting 

<!-- Convert joint states to /tf tranforms -->
<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>,
  
since we want to use our own URDF that includes the robotiq and realsense.

roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.100   kinematics_config:=$(rospack find ur_calibration)/etc/my_robot_calibration.yaml

for the robot bringup launch file, comment robot_state_publisher

(Go to UR5 screen) - follow directions there
a) Go to init screen
b) Power ON - Start - OK
b) Run Program
c) File - Load Program
d) mmnewer.urp
e) Play button (Terminal should read connected to reverse interface)
  
  
## 2. launch the gripper driver for arduino

roslaunch robotiq_2f_gripper_control gripper_bringup.launch

  -> in robotiq/robotiq_modbus_rtu/src/robotiq_modbus_rtu/comModbusRtu.py
1. chanage line 58: "print "Unable to connect to %s" % device" to "print ("Unable to connect to %s" % device)"
2. change line 76: "for i in range(0, len(data)/2):" to "for i in range(0, int(len(data)/2)):"

Also, check if joints match here in URDF file:


joints
-> in robotiq/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model_macro.xacro: 
 <xacro:macro name="inner_finger_joint" params="prefix fingerprefix">
    <joint name="${prefix}${fingerprefix}_inner_finger_joint" type="revolute">
      <origin xyz="0 0.0061 0.0471" rpy="0 0 0"/>
      <parent link="${prefix}${fingerprefix}_outer_finger" />
      <child link="${prefix}${fingerprefix}_inner_finger" />
      <axis xyz="-1 0 0" /> <!-- from <axis xyz="1 0 0" /> to <axis xyz="-1 0 0" /> -->
      <limit lower="0" upper="0.8757" velocity="2.0" effort="1000" />
      <mimic joint="${prefix}finger_joint" multiplier="1" offset="0" /> <!-- from multiplier="-1" to multiplier="1" -->
    </joint>
  </xacro:macro>
  
  
  

## 3. Moveit configuration launch

roslaunch ur5_with_robotiq_moveit_config ur5_gripper_moveit_planning_execution.launch 


RViz 
a) Fixed Frame - base_link
b) Add - Motion planning




## 4. RGBD camera realsense node with pointcloud

    roslaunch realsense2_camera rs_camera.launch filters:=pointcloud

## 5. Launch perception node for manipulation
    roslaunch simple_grasping basic_perception_dbg.launch 
    
-> create launch file
1. create "launch" folder in simple_grasping
2. create file "basic_perception_dbg.launch" under "launch" folder
3. in "basic_perception_dbg.launch": 
<launch>

  <!-- Start Perception -->
  <node name="basic_grasping_perception" pkg="simple_grasping" type="basic_grasping_perception" >
    <rosparam command="load" file="$(find simple_grasping)/config/simple_grasping.yaml" />
  </node>

</launch>
4. create folder "config"
5. create file "simple_grasping.yaml" under "launch" folder
5. in "simple_grasping.yaml": 


# yaml file for package simple_grasping with rosnode "basic_grasping_perception"
############################################################################
gripper:
  tool_to_planning_frame: 0.150    # should be 0.165 after fix the simple_grasping
  # finger tips are 195mm from wrist_roll_link, fingers are 60mm deep
  finger_depth: 0.020
  gripper_tolerance: 0.05
  approach:
    min: 0.145
    desired: 0.15
  retreat:
    min: 0.145
    desired: 0.15

use_debug: True


## 6. Run python file
  rosrun ur5_python_moveit_wrapper move_group_python_interface_tutorial.py
  

Summary of launch files: 

-------------------------------------------------- 
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.0.100 kinematics_config:=$(rospack find ur_calibration)/etc/my_robot_calibration.yaml

roslaunch robotiq_2f_gripper_control gripper_bringup.launch

roslaunch ur5_with_robotiq_moveit_config ur5_gripper_moveit_planning_execution.launch

roslaunch realsense2_camera rs_camera.launch filters:=pointcloud

roslaunch simple_grasping basic_perception_dbg.launch

rosrun ur5_python_moveit_wrapper move_group_python_interface_tutorial.py
--------------------------------------------------
  

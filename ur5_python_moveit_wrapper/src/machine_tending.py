#!/usr/bin/env python3


# Path planning for the arm resolution
# Author : Jeon Ho Kang


# Python 2/3 compatibility imports
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import JointConstraint 
import geometry_msgs.msg
import actionlib_msgs
from control_msgs.msg import GripperCommandAction, GripperCommandResult, GripperCommandGoal,GripperCommandActionGoal
from moveit_msgs.msg import Grasp
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
import actionlib
import time
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionResult
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi
from moveit_msgs.msg import Constraints
from std_msgs.msg import String, Int8
from moveit_commander.conversions import pose_to_list
import numpy as np
import tf
from tf import transformations


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
    


class movetodesiredposition(object) :
     def __init__(self):
        super(movetodesiredposition, self).__init__()
        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("pick_place_pipeline", anonymous=True)
        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        self.robot_traj_status = False
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        eef_name = "gripper"
        eef_group = moveit_commander.MoveGroupCommander(eef_name)
        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:

        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print('current state of the robot is : ')
        print(robot.get_current_state())
        print("")

        print("End of the moveit information")
        rospy.loginfo("Initializing machine tending")
        input("Press Enter to proceed : ")

        self.box_name = ""
        self.box_name2 = ""
        self.box_name3 = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.eef_group = eef_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.place_time = 1
        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
 
        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")

        pose_current = move_group.get_current_pose().pose
        print("The current pose is", pose_current) 


        # Grasp message generation 
        grasps = Grasp()
        grasps.pre_grasp_posture.joint_names = ['finger_joint']
        grasps.grasp_posture.joint_names = ['finger_joint']
        grasps.grasp_quality = 0.25
        grasps.pre_grasp_posture.points = [0.0]
        grasps.grasp_posture.points = [0.8]
        grasps.pre_grasp_approach.direction.vector = [0,0,1]
        grasps.pre_grasp_approach.desired_distance = 0.12
        grasps.pre_grasp_approach.min_distance = 0.0
        grasps.post_grasp_retreat.direction.vector = [0,0,-1]
        grasps.post_grasp_retreat.desired_distance = 0.08
        grasps.post_grasp_retreat.min_distance = 0.05
        grasps.pre_grasp_approach.direction.header.frame_id = 'wrist_3_link'
        grasps.post_grasp_retreat.direction.header.frame_id = 'wrist_3_link'
        self.grasps = grasps
        self.scene.remove_world_object()
        self.table_empty = False
        self.iteration = 1
        # print(grasps)
     

     def setup_scene(self):
        scene = self.scene
        self.goal.plan_grasps = False
        grasps = self.grasps
        # Insert support surface to the scene

     def look_pose(self):
        ## For the small sight (For the video of the shelf)
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        move_group.set_planning_time(3.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        # joint_goal = [-0.21123248735536748, -1.5190747419940394, 1.3441920280456543, -1.3246005217181605, -1.6159423033343714, -0.20117742220033819]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

     def scanning_mode(self):
        ## For the small sight (For the video of the shelf)
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        move_group.set_planning_time(3.0)
        move_group.set_max_velocity_scaling_factor(0.1)
        move_group.set_max_acceleration_scaling_factor(0.1)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        time.sleep(1)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        time.sleep(1)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        time.sleep(1)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


     def machine_tending_pose(self):
        #   x: 0.487740388675637
        #   y: -0.4094094524032348
        #   z: 0.44169172396390455
        # orientation: 
        #   x: -0.14927884652488957
        #   y: 0.9840614662834265
        #   z: -0.07972817760701974
        #   w: 0.054610202817557595
        ## For the small sight (For the video of the shelf)
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('RRTConnect')
        move_group.set_planning_time(3.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = "shoulder_pan_joint"
        # joint_constraint.position = 0.0  # should we use the current angle instead of "any valid"?
        joint_constraint.tolerance_above = pi   # in total 180°- we should keep <<360° to avoid 2 solutions for the same pose
        joint_constraint.tolerance_below = pi   # is this relative around position or absolute?
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape
        state = move_group.get_current_state()
        move_group.set_start_state(state)
        pose_goal.orientation.x = -0.14927884652488957
        pose_goal.orientation.y = 0.9840614662834265
        pose_goal.orientation.z = -0.07972817760701974
        pose_goal.orientation.w = 0.054610202817557595
        pose_goal.position.x =  0.487740388675637
        pose_goal.position.y = -0.4094094524032348
        pose_goal.position.z =  0.44169172396390455

        pose_current = move_group.get_current_pose()
        pose_current_pose = pose_current.pose
        # constraints = Constraints()
        # constraints.name = "base_constraint"
        # constraints.joint_constraints.header = 
        # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
        # constraints.joint_constraints.position = 0
        # constraints.joint_constraints.tolerance_below = -80
        # constraints.joint_constraints.tolerance_above = 80
        # constraints.joint_constraints.weight = 0.7
        # move_group.set_path_constraints(constraints)

        move_group.set_pose_target(pose_goal)
        # print('constraints :', constraints)
        ## Now, we call the planner to compute the plan and execute it.
        # move_group.go(plan, wait=True)
        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

     def start_pose(self):
        #   x: 0.24005759420789577
        #   y: -0.06246551676043084
        #   z: 0.8104004619886047
        # orientation: 
        #   x: -0.2985172448330094
        #   y: 0.9341914659432804
        #   z: -0.19503277188568577
        #   w: 0.011660934209577515

        ## For the small sight (For the video of the shelf)
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('RRTConnect')
        move_group.set_planning_time(3.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        joint_goal = move_group.get_current_joint_values()
        state = move_group.get_current_state()
        move_group.set_start_state(state)
        print("moving from", joint_goal)
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = "shoulder_pan_joint"
        # joint_constraint.position = 0.0  # should we use the current angle instead of "any valid"?
        joint_constraint.tolerance_above = pi   # in total 180°- we should keep <<360° to avoid 2 solutions for the same pose
        joint_constraint.tolerance_below = pi   # is this relative around position or absolute?
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape
        
        pose_goal.orientation.x = -0.2985172448330094
        pose_goal.orientation.y = 0.9341914659432804
        pose_goal.orientation.z = -0.19503277188568577
        pose_goal.orientation.w = 0.011660934209577515
        pose_goal.position.x =  0.24005759420789577
        pose_goal.position.y = -0.06246551676043084
        pose_goal.position.z =  0.8104004619886047

        pose_current = move_group.get_current_pose()
        pose_current_pose = pose_current.pose
        # constraints = Constraints()
        # constraints.name = "base_constraint"
        # constraints.joint_constraints.header = 
        # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
        # constraints.joint_constraints.position = 0
        # constraints.joint_constraints.tolerance_below = -80
        # constraints.joint_constraints.tolerance_above = 80
        # constraints.joint_constraints.weight = 0.7
        # move_group.set_path_constraints(constraints)

        move_group.set_pose_target(pose_goal)
        # print('constraints :', constraints)
        ## Now, we call the planner to compute the plan and execute it.
        # move_group.go(plan, wait=True)
        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()


        


     def end_pose(self):
#   x: 0.5950578229587558
#   y: 0.05762722752651399
#   z: 0.21306155127149337
# orientation: 
#   x: -0.6273646152996052
#   y: 0.7762798145986477
#   z: -0.03436922470595896
#   w: 0.05120591088711178
        ## For the small sight (For the video of the shelf)
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('RRTConnect')
        move_group.set_planning_time(3.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        joint_goal = move_group.get_current_joint_values()
        state = move_group.get_current_state()
        move_group.set_start_state(state)
        print("moving from", joint_goal)
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = "shoulder_pan_joint"
        # joint_constraint.position = 0.0  # should we use the current angle instead of "any valid"?
        joint_constraint.tolerance_above = pi/2   # in total 180°- we should keep <<360° to avoid 2 solutions for the same pose
        joint_constraint.tolerance_below = pi/2  # is this relative around position or absolute?
        pose_goal.orientation.x = -0.6273646152996052
        pose_goal.orientation.y = 0.7762798145986477
        pose_goal.orientation.z = -0.03436922470595896
        pose_goal.orientation.w = 0.05120591088711178
        pose_goal.position.x =  0.5950578229587558
        pose_goal.position.y = 0.05762722752651399
        pose_goal.position.z =  0.21306155127149337+0.01

        pose_current = move_group.get_current_pose()
        pose_current_pose = pose_current.pose
        # constraints = Constraints()
        # constraints.name = "base_constraint"
        # constraints.joint_constraints.header = 
        # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
        # constraints.joint_constraints.position = 0
        # constraints.joint_constraints.tolerance_below = -80
        # constraints.joint_constraints.tolerance_above = 80
        # constraints.joint_constraints.weight = 0.7
        # move_group.set_path_constraints(constraints)

        move_group.set_pose_target(pose_goal)
        # print('constraints :', constraints)
        ## Now, we call the planner to compute the plan and execute it.
        # move_group.go(plan, wait=True)
        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

     def middle_look_pose_large_table(self):
        ## For large table on industry day
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(1.0)
        move_group.set_max_velocity_scaling_factor(0.9)
        move_group.set_max_acceleration_scaling_factor(0.3)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        # joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        joint_goal = [-0.21032315889467412, -1.477865997944967, 0.6152205467224121, -0.5599568525897425, -1.6021764914142054, -0.3120043913470667]
        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


     def right_look_pose_large_table(self):
        ## For large table on industry day
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(1.0)
        move_group.set_max_velocity_scaling_factor(0.9)
        move_group.set_max_acceleration_scaling_factor(0.3)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        # joint_goal = [-0.6033108870135706, -1.6053898970233362, 0.4835667610168457, -0.7608488241778772, -1.5396769682513636, -0.47837621370424444]
        joint_goal = [0.8680226802825928, -1.415863339100973, 0.7293338775634766, -0.9470303694354456, -1.6433189550982874, 0.8256299495697021]

        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


     def left_look_pose_large_table(self):
        ## For large table on industry day
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(1.0)
        move_group.set_max_velocity_scaling_factor(0.9)
        move_group.set_max_acceleration_scaling_factor(0.3)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        # joint_goal = [0.4373485743999481, -1.310164753590719, 0.5260915756225586, -1.0633671919452112, -1.5239480177508753, -0.09342080751527959]
        joint_goal = [-0.0842974821673792, -1.0797084013568323, 0.5550117492675781, -1.1782687346087855, -1.565169636403219, -0.22676068941225225]
        move_group.go(joint_goal, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()    


     def grasp_posture(self):
        # Get gripper links
        grasps = self.grasps 
        eef_group = self.eef_group
        eeflinks = eef_group.get_end_effector_link()
        gripper_pose = eef_group.get_current_joint_values()
        print('current gripper pose :', gripper_pose)
        state = eef_group.get_current_state()
        eef_group.set_start_state(state)
        gripper_pose[0] = 0.8
        rospy.loginfo(gripper_pose)
        eef_group.go(gripper_pose, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        eef_group.stop()
        

     def pre_grasp_posture(self):
        # Get gripper links
        eef_group = self.eef_group
        grasps = self.grasps 
        eeflinks = eef_group.get_end_effector_link()
        state = eef_group.get_current_state()
        eef_group.set_start_state(state)
        gripper_pose = eef_group.get_current_joint_values()
        gripper_pose[0] = 0.0
        rospy.loginfo(gripper_pose)
        eef_group.go(gripper_pose, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        eef_group.stop()


    #  def pre_open(self):
    #     # Copy class variables to local variables to make the web tutorials more clear.
    #     # In practice, you should use the class variables directly unless you have a good
    #     # reason not to.
    #     move_group = self.move_group
    #     grasps = self.grasps
    #     # move_group.set_planner_id('AnytimePathShortening')
    #     move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
    #     move_group.set_planner_id('LIN')

    #     ## BEGIN_SUB_TUTORIAL plan_to_pose
    #     ##
    #     ## Planning to a Pose Goal
    #     ## ^^^^^^^^^^^^^^^^^^^^^^^
    #     ## We can plan a motion  for this group to a desired pose for the
    #     ## end-effector:
    #     pose_goal = geometry_msgs.msg.Pose()
    #     # pose of the tape

    #     pose_goal.orientation.x = 0.7172362562958459
    #     pose_goal.orientation.y = -0.6951907105171228
    #     pose_goal.orientation.z = -0.0436827629569143
    #     pose_goal.orientation.w = 0.01933506880949618
    #     pose_goal.position.x = 0.6871409420405031
    #     pose_goal.position.y = 0.006108689357482331
    #     pose_goal.position.z =  0.006108689357482331
        
    #     pose_current = move_group.get_current_pose()
    #     pose_current_pose = pose_current.pose
    #     # constraints = Constraints()
    #     # constraints.name = "base_constraint"
    #     # constraints.joint_constraints.header = 
    #     # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
    #     # constraints.joint_constraints.position = 0
    #     # constraints.joint_constraints.tolerance_below = -80
    #     # constraints.joint_constraints.tolerance_above = 80
    #     # constraints.joint_constraints.weight = 0.7
    #     # move_group.set_path_constraints(constraints)
    #     print(pose_current_pose) 
    #     move_group.set_pose_target(pose_goal)
    #     # print('constraints :', constraints)
    #     ## Now, we call the planner to compute the plan and execute it.
    #     # move_group.go(plan, wait=True)
    #     move_group.go(wait=True)
    #     # Calling `stop()` ensures that there is no residual movement
    #     move_group.stop()
    #     # It is always good to clear your targets after planning with poses.
    #     # Note: there is no equivalent function for clear_joint_value_targets()
    #     move_group.clear_pose_targets()

     def pre_open(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        grasps = self.grasps
        # move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        move_group.set_num_planning_attempts(10)
        move_group.set_goal_tolerance(0.01)

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape

        pose_goal.orientation.x = 0.7250307025166701
        pose_goal.orientation.y = -0.6880518954539065
        pose_goal.orientation.z = 0.029039156328577603
        pose_goal.orientation.w = 0.00847330928235808
        pose_goal.position.x =  0.6646784740105551
        pose_goal.position.y = -0.006195053083231175
        pose_goal.position.z =  0.1279936911502953
#           x: 0.6646784740105551
#   y: -0.006195053083231175
#   z: 0.1279936911502953
# orientation: 
#   x: 0.7250307025166701
#   y: -0.6880518954539065
#   z: 0.029039156328577603
#   w: 0.00847330928235808

        pose_current = move_group.get_current_pose()
        pose_current_pose = pose_current.pose
        # constraints = Constraints()
        # constraints.name = "base_constraint"
        # constraints.joint_constraints.header = 
        # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
        # constraints.joint_constraints.position = 0
        # constraints.joint_constraints.tolerance_below = -80
        # constraints.joint_constraints.tolerance_above = 80
        # constraints.joint_constraints.weight = 0.7
        # move_group.set_path_constraints(constraints)
        print(pose_current_pose) 
        move_group.set_pose_target(pose_goal)
        # print('constraints :', constraints)
        ## Now, we call the planner to compute the plan and execute it.
        # move_group.go(plan, wait=True)
        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
    
     def open(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group
            grasps = self.grasps 
            move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            move_group.set_planner_id('LIN')
            ## BEGIN_SUB_TUTORIAL plan_cartesian_path
            ##
            ## Cartesian Paths
            ## ^^^^^^^^^^^^^^^
            ## You can plan a Cartesian path directly by specifying a list of waypoints
            ## for the end-effector to go through. If executing  interactively in a     
            ## Python shell, set scale = 1.0.
            ##
            waypoints = []
            # move_group.set_pose_reference_frame(grasps.pre_grasp_approach.direction.header.frame_id)
            wpose = move_group.get_current_pose().pose
            wpose.position.x -= scale * 0.23

            waypoints.append(copy.deepcopy(wpose))


            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.001, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planning, not asking move_group to actually move the robot yet:
            return plan, fraction
    
     def approach(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group
            grasps = self.grasps 
            move_group.set_planning_pipeline_id("ompl")
            # move_group.set_planner_id('LIN')
            ## BEGIN_SUB_TUTORIAL plan_cartesian_path
            ##
            ## Cartesian Paths
            ## ^^^^^^^^^^^^^^^
            ## You can plan a Cartesian path directly by specifying a list of waypoints
            ## for the end-effector to go through. If executing  interactively in a     
            ## Python shell, set scale = 1.0.
            ##
            waypoints = []
            # move_group.set_pose_reference_frame(grasps.pre_grasp_approach.direction.header.frame_id)
            wpose = move_group.get_current_pose().pose
            if grasps.pre_grasp_approach.direction.vector[0] == 1:
                wpose.position.x -= scale * grasps.pre_grasp_approach.desired_distance # move in x direction
            elif grasps.pre_grasp_approach.direction.vector[1] == 1:
                wpose.position.y -= scale * grasps.pre_grasp_approach.desired_distance # move in y direction
            elif grasps.pre_grasp_approach.direction.vector[2] == 1:
                wpose.position.z -= scale * grasps.pre_grasp_approach.desired_distance # move in y direction
            elif grasps.pre_grasp_approach.direction.vector[0] == -1:
                wpose.position.x += scale * grasps.pre_grasp_approach.desired_distance # move in y direction
            elif grasps.pre_grasp_approach.direction.vector[1] == -1:
                wpose.position.y += scale * grasps.pre_grasp_approach.desired_distance # move in y direction
            elif grasps.pre_grasp_approach.direction.vector[2] == -1:
                wpose.position.z += scale * grasps.pre_grasp_approach.desired_distance # move in y direction
            # otherwise default to +z direction
            else:
                wpose.position.z -= scale * grasps.pre_grasp_approach.desired_distance # move in y direction

            waypoints.append(copy.deepcopy(wpose))


            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.tta
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.001, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planning, not asking move_group to actually move the robot yet:
            return plan, fraction

     def retreat(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group
            grasps = self.grasps
            move_group.set_planning_pipeline_id("ompl")
            move_group.set_planner_id('AnytimePathShortening')
            move_group.set_planning_time(3.0)
            move_group.set_max_velocity_scaling_factor(0.9)
            move_group.set_max_acceleration_scaling_factor(0.2)
            joint_goal = move_group.get_current_joint_values()
            ## BEGIN_SUB_TUTORIAL plan_cartesian_path
            ##
            ## Cartesian Paths
            ## ^^^^^^^^^^^^^^^
            ## You can plan a Cartesian path directly by specifying a list of waypoints
            ## for the end-effector to go through. If executing  interactively in a     
            ## Python shell, set scale = 1.0.
            ##
            waypoints = []

            wpose = move_group.get_current_pose().pose
            # move_group.set_pose_reference_frame(grasps.post_grasp_retreat.direction.header.frame_id)
            if grasps.post_grasp_retreat.direction.vector[0] == 1:
                wpose.position.x -= scale * grasps.post_grasp_retreat.desired_distance # move in x direction
            elif grasps.post_grasp_retreat.direction.vector[1] == 1:
                wpose.position.y -= scale * grasps.post_grasp_retreat.desired_distance # move in y direction
            elif grasps.post_grasp_retreat.direction.vector[2] == 1:
                wpose.position.z -= scale * grasps.post_grasp_retreat.desired_distance # move in y direction
            elif grasps.post_grasp_retreat.direction.vector[0] == -1:
                wpose.position.x += scale * grasps.post_grasp_retreat.desired_distance # move in y direction
            elif grasps.post_grasp_retreat.direction.vector[1] == -1:
                wpose.position.y += scale * grasps.post_grasp_retreat.desired_distance # move in y direction
            elif grasps.post_grasp_retreat.direction.vector[2] == -1:
                wpose.position.z += scale * grasps.post_grasp_retreat.desired_distance # move in y direction
            # otherwise default to -z direction
            else:
                wpose.position.z += scale * grasps.post_grasp_retreat.desired_distance # move in z direction

            waypoints.append(copy.deepcopy(wpose))


            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planni  ng, not asking move_group to actually move the robot yet:
            return plan, fraction


     def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)

        ## END_SUB_TUTORIAL

     def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)    
        return -1
     
     def go_to_pose_goal(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        grasps = self.grasps
        # move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.9)
        move_group.set_max_acceleration_scaling_factor(0.3)
        move_group.set_num_planning_attempts(10)
        move_group.set_goal_tolerance(0.01)

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape

        pose_goal.orientation.x = grasps.grasp_pose.pose.orientation.x
        pose_goal.orientation.y = grasps.grasp_pose.pose.orientation.y
        pose_goal.orientation.z = grasps.grasp_pose.pose.orientation.z
        pose_goal.orientation.w = grasps.grasp_pose.pose.orientation.w
        pose_goal.position.x = grasps.grasp_pose.pose.position.x
        pose_goal.position.y = grasps.grasp_pose.pose.position.y
        pose_goal.position.z =  grasps.grasp_pose.pose.position.z
        
        pose_current = move_group.get_current_pose()
        pose_current_pose = pose_current.pose
        # constraints = Constraints()
        # constraints.name = "base_constraint"
        # constraints.joint_constraints.header = 
        # # constraints.joint_constraints.joint_name = 'shoulder_pan_joint'
        # constraints.joint_constraints.position = 0
        # constraints.joint_constraints.tolerance_below = -80
        # constraints.joint_constraints.tolerance_above = 80
        # constraints.joint_constraints.weight = 0.7
        # move_group.set_path_constraints(constraints)
        print(pose_current_pose) 
        move_group.set_pose_target(pose_goal)
        # print('constraints :', constraints)
        ## Now, we call the planner to compute the plan and execute it.
        # move_group.go(plan, wait=True)
        move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

     def place_joint(self):
        move_group = self.move_group
        move_group.set_planning_pipeline_id("ompl")
        move_group.set_planner_id('EST')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.9)
        move_group.set_max_acceleration_scaling_factor(0.2)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        # joint_goal = [-2.9814820925342005, -1.5915573279010218, 1.7439374923706055, -1.8465951124774378, -1.5855477491961878, 0.018337737768888474]
        joint_goal = [-1.1302769819842737, -1.4423883597003382, 1.4548125267028809, -1.5902708212481897, -1.6805809179889124, 0.5314168930053711]
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
     def tool_pre_close1(self):

        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = -0.008399103228774267
        pose_goal.orientation.y = 0.992489608034789
        pose_goal.orientation.z = -0.11499678925553322
        pose_goal.orientation.w = 0.040860389973931134
        pose_goal.position.x = 0.43589918075947276
        pose_goal.position.y = -0.3101996023176584
        pose_goal.position.z = 0.18721699784908308 
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

     def tool_close1(self):

        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')

        # move_group.set_planner_id('AplnytimePathShortening')
        # move_group.set_planning_time(10.0)
        # move_group.set_max_velocity_scaling_factor(0.2)
        # move_group.set_max_acceleration_scaling_factor(0.1)
        # move_group.set_num_planning_attempts(10)
        # move_group.allow_replanning(6)
        # move_group.set_pose_reference_frame('wrist_3_link')
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.x = -0.6543331288968377

        # pose_goal.orientation.y = -0.7536531150227925

        # pose_goal.orientation.z = 0.042577788467737306

        # pose_goal.orientation.w = 0.04519148784574925

        # pose_goal.position.x = -0.47601423580396723

        # pose_goal.position.y = -0.1842329495574632

        # pose_goal.position.z = 0.2525356029470456
#   x: -0.5322842000501384
#   y: -0.21128624216261904
#   z: 0.46619695305822473
# orientation: 
#   x: -0.6036021012319646
#   y: -0.7963629444894539
#   z: 0.021884081046366785
#   w: 0.031490491091759144

#   x: 0.3325077556729014
#   y: 0.10962579053298091
#   z: 0.5055173333728407
# orientation: 
#   x: -0.7611227812270727
#   y: 0.6462109301699175
#   z: 0.04934474877722504



        # x: 0.42407425019770056
        # y: -0.31978050785288314
        # z: 0.17541990346357747
        # orientation: 
        # x: 0.011346767514879482
        # y: -0.9678872093750523
        # z: 0.2511278597000758
        # w: 0.0006315675702187416

#   w: 0.025858101133071652 
        pose_goal.orientation.x = -0.008399103228774267
        pose_goal.orientation.y = 0.992489608034789
        pose_goal.orientation.z = -0.11499678925553322
        pose_goal.orientation.w = 0.040860389973931134
        pose_goal.position.x = 0.43589918075947276
        pose_goal.position.y = -0.3101996023176584 - 0.33
        pose_goal.position.z = 0.18721699784908308 
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
     def tool_pre_close2(self):

        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')

        # move_group.set_planner_id('AplnytimePathShortening')
        # move_group.set_planning_time(10.0)
        # move_group.set_max_velocity_scaling_factor(0.2)
        # move_group.set_max_acceleration_scaling_factor(0.1)
        # move_group.set_num_planning_attempts(10)
        # move_group.allow_replanning(6)
        # move_group.set_pose_reference_frame('wrist_3_link')
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.x = -0.6543331288968377

        # pose_goal.orientation.y = -0.7536531150227925

        # pose_goal.orientation.z = 0.042577788467737306

        # pose_goal.orientation.w = 0.04519148784574925

        # pose_goal.position.x = -0.47601423580396723

        # pose_goal.position.y = -0.1842329495574632

        # pose_goal.position.z = 0.2525356029470456

#   x: 0.43589918075947276
#   y: -0.3101996023176584
#   z: 0.18721699784908308
# orientation: 
#   x: -0.008399103228774267
#   y: 0.992489608034789
#   z: -0.11499678925553322
#   w: 0.040860389973931134

#   w: 0.025858101133071652 
        pose_goal.orientation.x = -0.008399103228774267
        pose_goal.orientation.y = 0.992489608034789
        pose_goal.orientation.z = -0.11499678925553322
        pose_goal.orientation.w = 0.040860389973931134
        pose_goal.position.x = 0.43589918075947276
        pose_goal.position.y = -0.2801996023176584
        pose_goal.position.z = 0.13721699784908308 
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

     def tool_close2(self):

        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')

        # move_group.set_planner_id('AplnytimePathShortening')
        # move_group.set_planning_time(10.0)
        # move_group.set_max_velocity_scaling_factor(0.2)
        # move_group.set_max_acceleration_scaling_factor(0.1)
        # move_group.set_num_planning_attempts(10)
        # move_group.allow_replanning(6)
        # move_group.set_pose_reference_frame('wrist_3_link')
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.x = -0.6543331288968377

        # pose_goal.orientation.y = -0.7536531150227925

        # pose_goal.orientation.z = 0.042577788467737306

        # pose_goal.orientation.w = 0.04519148784574925

        # pose_goal.position.x = -0.47601423580396723

        # pose_goal.position.y = -0.1842329495574632

        # pose_goal.position.z = 0.2525356029470456
#   x: -0.5322842000501384
#   y: -0.21128624216261904
#   z: 0.46619695305822473
# orientation: 
#   x: -0.6036021012319646
#   y: -0.7963629444894539
#   z: 0.021884081046366785
#   w: 0.031490491091759144

#   x: 0.3325077556729014
#   y: 0.10962579053298091
#   z: 0.5055173333728407
# orientation: 
#   x: -0.7611227812270727
#   y: 0.6462109301699175
#   z: 0.04934474877722504



        # x: 0.42407425019770056
        # y: -0.31978050785288314
        # z: 0.17541990346357747
        # orientation: 
        # x: 0.011346767514879482
        # y: -0.9678872093750523
        # z: 0.2511278597000758
        # w: 0.0006315675702187416

#   w: 0.025858101133071652 

        pose_goal.orientation.x = -0.008399103228774267
        pose_goal.orientation.y = 0.992489608034789
        pose_goal.orientation.z = -0.11499678925553322
        pose_goal.orientation.w = 0.040860389973931134
        pose_goal.position.x = 0.43589918075947276
        pose_goal.position.y = -0.2801996023176584 - 0.35
        pose_goal.position.z = 0.13721699784908308 
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
     def place(self):
        
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')

        # move_group.set_planner_id('AplnytimePathShortening')
        # move_group.set_planning_time(5.0)
        # move_group.set_max_velocity_scaling_factor(0.2)
        # move_group.set_max_acceleration_scaling_factor(0.1)
        # move_group.set_num_planning_attempts(10)
        # move_group.allow_replanning(6)
        # move_group.set_pose_reference_frame('wrist_3_link')
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.x = -0.6543331288968377

        # pose_goal.orientation.y = -0.7536531150227925

        # pose_goal.orientation.z = 0.042577788467737306

        # pose_goal.orientation.w = 0.04519148784574925

        # pose_goal.position.x = -0.47601423580396723

        # pose_goal.position.y = -0.1842329495574632

        # pose_goal.position.z = 0.2525356029470456
#   x: -0.5322842000501384
#   y: -0.21128624216261904
#   z: 0.46619695305822473
# orientation: 
#   x: -0.6036021012319646
#   y: -0.7963629444894539
#   z: 0.021884081046366785
#   w: 0.031490491091759144

#   x: 0.3325077556729014
#   y: 0.10962579053298091
#   z: 0.5055173333728407
# orientation: 
#   x: -0.7611227812270727
#   y: 0.6462109301699175
#   z: 0.04934474877722504
#   w: 0.025858101133071652 
        if self.place_time > 4:
            offset = self.place_time - 4
        else:
            offset = self.place_time
        print("The offset is " , offset)
        if offset == 1: 
            pose_goal.orientation.x = 0.7130321552400869
            pose_goal.orientation.y = -0.6571253452755551
            pose_goal.orientation.z = 0.003570895297731319
            pose_goal.orientation.w = 0.033483964971325796
            pose_goal.position.x = 0.4270080275169177
            pose_goal.position.y = -0.5187426447439903
            pose_goal.position.z = 0.41572822953833 
        elif offset == 2: 
            pose_goal.orientation.x = 0.7130321552400869
            pose_goal.orientation.y = -0.6571253452755551 
            pose_goal.orientation.z = 0.003570895297731319
            pose_goal.orientation.w = 0.033483964971325796
            pose_goal.position.x = 0.4270080275169177
            pose_goal.position.y = -0.5187426447439903 - 0.05
            pose_goal.position.z = 0.41572822953833 
        elif offset == 3:
            pose_goal.orientation.x = 0.7130321552400869 
            pose_goal.orientation.y = -0.6571253452755551 
            pose_goal.orientation.z = 0.003570895297731319
            pose_goal.orientation.w = 0.033483964971325796
            pose_goal.position.x = 0.4270080275169177 - 0.05
            pose_goal.position.y = -0.5187426447439903
            pose_goal.position.z = 0.41572822953833 
        elif offset == 4:
            pose_goal.orientation.x = 0.7130321552400869 
            pose_goal.orientation.y = -0.6571253452755551 
            pose_goal.orientation.z = 0.003570895297731319
            pose_goal.orientation.w = 0.033483964971325796
            pose_goal.position.x = 0.4270080275169177+0.06
            pose_goal.position.y = -0.5187426447439903
            pose_goal.position.z = 0.41572822953833 
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
    
     def place_approach(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group
            grasps = self.grasps 
            move_group.set_planning_pipeline_id("ompl")
            # move_group.set_planner_id('LIN')
            ## BEGIN_SUB_TUTORIAL plan_cartesian_path
            ##
            ## Cartesian Paths
            ## ^^^^^^^^^^^^^^^
            ## You can plan a Cartesian path directly by specifying a list of waypoints
            ## for the end-effector to go through. If executing  interactively in a     
            ## Python shell, set scale = 1.0.
            ##
            waypoints = []
            # move_group.set_pose_reference_frame(grasps.pre_grasp_approach.direction.header.frame_id)
            wpose = move_group.get_current_pose().pose
            desired_distance = 0.18
            if self.place_time > 4:
                desired_distance = 0.26
            wpose.position.z -= scale * desired_distance # move in y direction

            waypoints.append(copy.deepcopy(wpose))


            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.tta
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.001, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planning, not asking move_group to actually move the robot yet:
            return plan, fraction

     def give_the_tool(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
         # move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        move_group.set_planner_id('LIN')
        # move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        move_group.set_num_planning_attempts(10)
        move_group.allow_replanning(6)

        # move_group.set_pose_reference_frame('wrist_3_link')
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = 0.3704324798640179

        pose_goal.orientation.y = -0.7281174889088765

        pose_goal.orientation.z =  0.4966418367544429

        pose_goal.orientation.w = 0.2932091168296093

        pose_goal.position.x = 0.7282852321856831

        pose_goal.position.y = -0.3713837474580142

        pose_goal.position.z = 0.4777866644182047


# x: 0.7282852321856831
#   y: -0.3713837474580142
#   z: 0.4777866644182047
# orientation: 
#   x: 0.3704324798640179
#   y: -0.7281174889088765
#   z: 0.4966418367544429
#   w: 0.2932091168296093

#   x: 0.8408391939087037
#   y: -0.22524865491664653
#   z: 0.5349457603030249
# orientation: 
#   x: -0.3862576256331682
#   y: 0.5502874409507793
#   z: -0.6230185375925582
#   w: 0.39979579886249444


        print(move_group.get_pose_reference_frame())
        pose_current = move_group.get_current_pose().pose
        print(pose_current) 
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
    

     def alternative_start_pose(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(20.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        move_group.set_num_planning_attempts(10)
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = 0.8206012066753539
        
        pose_goal.orientation.y = -0.4837091680837061

        pose_goal.orientation.z =  0.2657988435990258

        pose_goal.orientation.w = 0.14829050898875104

        pose_goal.position.x = 0.11650065622629119

        pose_goal.position.y = 0.3258799926885096

        pose_goal.position.z = 0.5554785823629577
#   x: 0.11650065622629119
#   y: 0.3258799926885096
#   z: 0.5554785823629577
# orientation: 
#   x: 0.8206012066753539
#   y: -0.4837091680837061
#   z: 0.2657988435990258
#   w: 0.14829050898875104



        print(move_group.get_pose_reference_frame())
        pose_current = move_group.get_current_pose().pose
        print(pose_current) 
        # Now, we call the planner to compute the plan and execute it.
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        ## END_SUB_TUTORIAL
        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
    


     def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Received
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL

     def add_table(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.04      # above the panda_hand frame
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(3, 3, 0.09))

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

     def add_pick_object(self, timeout =4):
         # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.x = 0.5703358071795699
        box_pose.pose.orientation.y = 0.8213533338510772
        box_pose.pose.orientation.z  = 0.0002819667169430619
        box_pose.pose.orientation.w = 0.009782050670480892
        box_pose.pose.position.x = -0.6283562498937836
        box_pose.pose.position.y = 0.06596562538431887
        box_pose.pose.position.z = 0.0

        # box_pose.pose.orientation.x = 
        # box_pose.pose.orientation.y =  
        # box_pose.pose.orientation.z =
              # above the panda_hand frame
        box_name = "pick_box"
        scene.add_box(box_name, box_pose, size=(0.02, 0.06,0.08))

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name = box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)


     def add_wall(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name2 = self.box_name2
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL add_boxplantherobot.go_to_pose_goal2()Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.03      # above the panda_hand frame
        box_pose.pose.position.y = -0.25   # above the panda_hand frame
  
        box_name2 = "Wall"
        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name2 = box_name2
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

     def add_top(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name2 = self.box_name3
        scene = self.scene
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.7     # above the panda_hand frame

        box_name3 = "box3"
        scene.add_box(box_name3, box_pose, size=(3, 3, 0.04))

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name3 = box_name3
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)
        
     def attach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = 'pick_object0'
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link

        ## BEGIN_SUB_TUTORIAL attach_object
        ##
        ## Attaching Objects to the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
        ## robot be able to touch them without the planning scene reporting the contact as a
        ## collision. By adding link names to the ``touch_links`` array, we are telling the
        ## planning scene to ignore collisions between those links and the box. For the Panda
        ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        ## you should change this value to the name of your end effector group name.
        grasping_group = "gripper"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links='wirst_3_link')
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout
        )

     def detach_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene
        eef_link = self.eef_link

        ## BEGIN_SUB_TUTORIAL detach_object
        ##
        ## Detaching Objects from the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can also detach and remove the object from the planning scene:
        scene.remove_attached_object(eef_link, name=box_name)
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

     def remove_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL remove_object
        ##
        ## Removing Objects from the Planning Scene', box_size.dimensions[0])\
        scene.remove_world_object(name='pick_object0')
        scene.remove_world_object(name='pick_object1')
        scene.remove_world_object(name='pick_object2')
        scene.remove_world_object(name='pick_object3')
        scene.remove_world_object(name='pick_object4')
        scene.remove_world_object(name='pick_object5')
        scene.remove_world_object(name='pick_object6')
        scene.remove_world_object(name='pick_object7')
        scene.remove_world_object(name='pick_object8')
        scene.remove_world_object(name='pick_object9')
        scene.remove_world_object(name='pick_object10')
        scene.remove_world_object(name='pick_object11')

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )
     def trajectory_callback(self, msg):
        # print the actual message in its raw format
        trajectory_status = msg.status.status
        # Set up goal
        if trajectory_status == 3:
            self.robot_traj_status = True
        else:
            self.robot_traj_status = False

     def close_gripper(self):
        sub = rospy.Subscriber('/scaled_pos_joint_traj_controller/follow_joint_trajectory/result', FollowJointTrajectoryActionResult, self.trajectory_callback)
        

def pick():
    try:
        print("Welcome")


        plantherobot = movetodesiredposition()
        plantherobot.scanning_mode()



        ## Machine Tending for 3D printer 
        ## Method of adding the table to avoid going down
        # plantherobot.add_table()
        # plantherobot.pre_grasp_posture()
        # time.sleep(2)
        # plantherobot.start_pose()
        # input('next step: ')
        # plantherobot.machine_tending_pose()
        # time.sleep(2)
        # plantherobot.grasp_posture()
        # plantherobot.start_pose()
        # plantherobot.end_pose()
        # time.sleep(1)
        # plantherobot.pre_grasp_posture()
        # plantherobot.start_pose()
        ### End of Machine Tending for 3D printer
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
if __name__ == "__main__":
    pick()

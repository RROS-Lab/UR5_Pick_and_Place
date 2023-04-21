#!/usr/bin/env python3


# Path planning for the arm resolution
# Author : Jeon Ho Kang


# Python 2/3 compatibility imports

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import actionlib_msgs
from control_msgs.msg import GripperCommandAction, GripperCommandResult, GripperCommandGoal,GripperCommandActionGoal
from moveit_msgs.msg import Grasp
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
import actionlib

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np
from tf import transformations


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
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        eef_name = "gripper"
        eef_group = moveit_commander.MoveGroupCommander(eef_name)
        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

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

        print("End of the moveit information")
        rospy.loginfo("Initializing Pick and Place pipeline")
        input("Press Enter to proceed : ")

        self.box_name = ""
        self.box_name2 = ""
        self.box_name3 = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.eef_group = eef_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        joint_goal = [-0.0008257071124475601, -1.9525201956378382, 0.9400134086608887, -0.9726727644549769, -1.65515643755068, 0.017052054405212402]
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
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
        ## END_SUB_TUTORIAL
        pose_current = move_group.get_current_pose().pose
        print("The current pose is", pose_current) 
        ## Input gripper profile
        max_gripper_opening = 0.05



        # Grasp message generation 
        grasps = Grasp()
        grasps.pre_grasp_posture.joint_names = ['left_finger_joint','right_finger_joint']
        grasps.grasp_posture.joint_names = ['left_finger_joint','right_finger_joint']
        grasps.grasp_quality = 0.25
        grasps.pre_grasp_posture.points = [0.0, 0.0]
        grasps.grasp_posture.points = [0.024, 0.024]
        grasps.pre_grasp_approach.direction.vector = [0,0,1]
        # grasps.pre_grasp_approach.desired_distance = 0.053
        grasps.pre_grasp_approach.min_distance = 0.05
        grasps.post_grasp_retreat.direction.vector = [0,0,-1]
        grasps.post_grasp_retreat.desired_distance = 0.05
        grasps.post_grasp_retreat.min_distance = 0.05
        grasps.pre_grasp_approach.direction.header.frame_id = 'wrist_3_link'
        grasps.post_grasp_retreat.direction.header.frame_id = 'wrist_3_link'

        self.grasps = grasps
        # print(grasps)

        scene.remove_world_object()
        find_topic = "basic_grasping_perception/find_objects"
        rospy.loginfo("Waiting for %s..." % find_topic)
        self.find_client = actionlib.SimpleActionClient(
            find_topic, FindGraspableObjectsAction)
        self.find_client.wait_for_server()
        goal = FindGraspableObjectsGoal()
        goal.plan_grasps = False
        self.find_client.send_goal(goal)
        self.find_client.wait_for_result()
        find_result = self.find_client.get_result()
        # Insert support surface to the scene
        for obj in find_result.support_surfaces:
            # extend surface to floor, and make wider since we have narrow field of view
            table_pose = geometry_msgs.msg.PoseStamped()
            table_pose.header.frame_id = 'base_link'
            table_pose.pose = obj.primitive_poses[0]
            height = obj.primitive_poses[0].position.z
            obj.primitives[0].dimensions = [obj.primitives[0].dimensions[0],
                                            1.5,  # wider
                                            obj.primitives[0].dimensions[2]-0.025]
            obj.primitive_poses[0].position.z += -height/2.6
            print('TABLE_POSITION Z', obj.primitive_poses[0].position.z)
            table_size = obj.primitives[0].dimensions
            print('pose primmitive :', obj.primitive_poses[0])
            # add to scene
            self.scene.add_box('support_surface',
                                         table_pose,
                                         size = (table_size[0],table_size[1],table_size[2]))
        # insert objects to scene
        objects = list()
        idx = -1            
        for obj in find_result.objects:
            idx += 1
            obj.object.name = "pick_object%d" % idx
            box_pose = geometry_msgs.msg.PoseStamped()
            box_pose.header.frame_id = "base_link"
            box_pose.pose = obj.object.primitive_poses[0]
            box_size = obj.object.primitives[0]
            print('MF', box_size)
            self.scene.add_box(obj.object.name,
                                         box_pose,
                                         size = (box_size.dimensions[0],box_size.dimensions[1],box_size.dimensions[2]))
            print("Object pose: ", box_pose)

        gripper_translation = [-0.04,0.01,0.05]       
        if box_size.dimensions[0] > max_gripper_opening-0.01:
            q_r = transformations.quaternion_from_euler(pi/4,pi,3*pi/2)
            print('gripper rotated by half pi')
        else:
            q_r = transformations.quaternion_from_euler(pi,0,0)

        if box_size.dimensions[2] > 0.05:
            gripper_translation = [-0.02,0,0.1]  
            q_r = transformations.quaternion_from_euler(0,pi,3*pi/2)
            grasps.pre_grasp_approach.desired_distance = 0.03
            print('Higher object pick')
        if box_size.dimensions[2] > 0.15:
            gripper_translation = [-0.1,0,0.1]  
            q_r = transformations.quaternion_from_euler(-pi/2,pi,3*pi/2)
            grasps.pre_grasp_approach.desired_distance = 0.03
            grasps.pre_grasp_approach.direction.vector = [1,0,0]
            print('Higher object lateralpick')

        q_org = [box_pose.pose.orientation.x, box_pose.pose.orientation.y, box_pose.pose.orientation.z, box_pose.pose.orientation.w]
        new_quaternion = transformations.quaternion_multiply(q_org,q_r)
        print(new_quaternion)
        grasps.grasp_pose.pose.orientation
        grasps.grasp_pose.pose.position.x = box_pose.pose.position.x + gripper_translation[0]
        grasps.grasp_pose.pose.position.y = box_pose.pose.position.y + gripper_translation[1]
        grasps.grasp_pose.pose.position.z =  box_pose.pose.position.z+ gripper_translation[2]
        grasps.grasp_pose.pose.orientation.x = new_quaternion[0]
        grasps.grasp_pose.pose.orientation.y = new_quaternion[1]
        grasps.grasp_pose.pose.orientation.z = new_quaternion[2]
        grasps.grasp_pose.pose.orientation.w = new_quaternion[3]
        print("the new grasp is", grasps.grasp_pose.pose)
        grasps.pre_grasp_approach.desired_distance = grasps.grasp_pose.pose.position.z+(box_size.dimensions[2])
        if box_size.dimensions[2] > 0.05:
            grasps.pre_grasp_approach.desired_distance = 0.08
            print('Higher object pick')
        print("approach distance is ", grasps.pre_grasp_approach.desired_distance)
  
    #  def start_state(self):
    #     # Copy class variables to local variables to make the web tutorials more clear.
    #     # In practice, you should use the class variables directly unless you have a good
    #     # reason not to.
    #     move_group = self.move_group

        





     def grasp_posture(self):
        # Get gripper links
        grasps = self.grasps 
        eef_group = self.eef_group
        eeflinks = eef_group.get_end_effector_link()
        gripper_pose = eef_group.get_current_joint_values()
        gripper_pose[0] = grasps.grasp_posture.points[0] # close left finger
        gripper_pose[1] = grasps.grasp_posture.points[1] # close right finger
        rospy.loginfo(gripper_pose)
        eef_group.go(gripper_pose, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        eef_group.stop()
        
     def pre_grasp_posture(self):
        # Get gripper links
        eef_group = self.eef_group
        grasps = self.grasps 
        eeflinks = eef_group.get_end_effector_link()
        gripper_pose = eef_group.get_current_joint_values()
        gripper_pose[0] = grasps.pre_grasp_posture.points[0] # open left finger
        gripper_pose[1] = grasps.pre_grasp_posture.points[1] # open right finger
        rospy.loginfo(gripper_pose)
        eef_group.go(gripper_pose, wait=True)
        # Calling ``stop()`` ensures that there is no residual movement
        eef_group.stop()
     
    
     def approach(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group
            grasps = self.grasps 

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
            # for this tutorial.
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
        move_group.set_planner_id('AnytimePathShortening')
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

        pose_goal.orientation.x = grasps.grasp_pose.pose.orientation.x
        pose_goal.orientation.y = grasps.grasp_pose.pose.orientation.y
        pose_goal.orientation.z = grasps.grasp_pose.pose.orientation.z
        pose_goal.orientation.w = grasps.grasp_pose.pose.orientation.w
        pose_goal.position.x = grasps.grasp_pose.pose.position.x
        pose_goal.position.y = grasps.grasp_pose.pose.position.y
        pose_goal.position.z =  grasps.grasp_pose.pose.position.z



        pose_current = move_group.get_current_pose().pose
        print(pose_current) 

        move_group.set_pose_target(pose_goal)
        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
     def place_joint(self):
        move_group = self.move_group
        
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(10.0)
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.set_max_acceleration_scaling_factor(0.1)
        move_group.set_num_planning_attempts(10)
        move_group.allow_replanning(6)
        joint_goal = move_group.get_current_joint_values()
        print("moving from", joint_goal)
        joint_goal = [-2.9814820925342005, -1.5915573279010218, 1.7439374923706055, -1.8465951124774378, -1.5855477491961878, 0.018337737768888474]
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        
     def place(self):
        move_group.set_planning_time(3.0)
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planner_id('AnytimePathShortening')
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
        pose_goal.orientation.x = -0.6543331288968377

        pose_goal.orientation.y = -0.7536531150227925

        pose_goal.orientation.z = 0.042577788467737306

        pose_goal.orientation.w = 0.04519148784574925

        pose_goal.position.x = -0.47601423580396723

        pose_goal.position.y = -0.1842329495574632

        pose_goal.position.z = 0.2525356029470456
#  x: -0.47601423580396723
#   y: -0.1842329495574632
#   z: 0.2525356029470456
# orientation: 
#   x: -0.6543331288968377
#   y: -0.7536531150227925
#   z: 0.042577788467737306
#   w: 0.04519148784574925

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


     def give_the_tool(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planner_id('AnytimePathShortening')
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
        pose_goal.orientation.x = -0.3862576256331682

        pose_goal.orientation.y = 0.5502874409507793

        pose_goal.orientation.z =  -0.6230185375925582

        pose_goal.orientation.w = 0.39979579886249444

        pose_goal.position.x = 0.8408391939087037

        pose_goal.position.y = -0.22524865491664653

        pose_goal.position.z = 0.5349457603030249

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
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(name='pick_object0')
        scene.remove_world_object(name='pick_object1')
        scene.remove_world_object(name='pick_object2')

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )

        
def pick():
    try:
        print("Welcome")
        plantherobot = movetodesiredposition()
        # # plantherobot.add_wall()
        plantherobot.add_table()
        # plantherobot.add_pick_object()
        plantherobot.pre_grasp_posture()
        plantherobot.go_to_pose_goal()
        plantherobot.remove_box()
        approach, fraction = plantherobot.approach()
        plantherobot.display_trajectory(approach)
        plantherobot.execute_plan(approach)
        input('Hold')
        plantherobot.grasp_posture()
        input('Hold')
        retreat, fraction = plantherobot.retreat()
        plantherobot.display_trajectory(retreat)
        plantherobot.execute_plan(retreat)
        # plantherobot.attach_box()
        # plantherobot.give_the_tool()
        plantherobot.place_joint()
        approach, fraction = plantherobot.approach()
        plantherobot.display_trajectory(approach)
        plantherobot.execute_plan(approach)
        plantherobot.pre_grasp_posture()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    pick()
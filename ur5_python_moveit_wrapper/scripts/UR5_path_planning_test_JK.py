#!/usr/bin/env python3


# Path planning for the arm resolution
# Author : Jeon Ho Kang


# Python 2/3 compatibility imports
from __future__ import print_function
from ast import JoinedStr
from distutils.filelist import glob_to_re
from pickletools import uint8
from re import M
from threading import get_native_id
from xml.sax.handler import property_declaration_handler
from six.moves import input
from std_msgs.msg import Int8



import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import actionlib_msgs



import copy
import actionlib
import rospy

from math import sin, cos
from moveit_python.geometry import rotate_pose_msg_by_euler_angles

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint



try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class movetodesiredposition(object) :
     def __init__(self):
        super(movetodesiredposition, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("pick_and_place_pipeline", anonymous=True)

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


        self.box_name = ""
        self.box_name2 = ""
        self.box_name3 = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        ## END_SUB_TUTORIAL

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


         
     def plan_cartesian_path(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group

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
            #wpose.position.z -= scale * 0.1  # First move up (z)
            #wpose.position.y += scale * 0.2  # and sideways (y)
           # waypoints.append(copy.deepcopy(wpose))

            wpose.position.x += scale * 0.3  # Second move forward/backwards in (x)
            wpose.position.z -= scale * 0.4  # First move up (z)
            wpose.position.y -= scale * 0.1  # and sideways (y)
            waypoints.append(copy.deepcopy(wpose))

            wpose.position.x -= scale * 0.3  # Second move forward/backwards in (x)
            wpose.position.z += scale * 0.4  # First move up (z)
            wpose.position.y += scale * 0.1  # and sideways (y)
            waypoints.append(copy.deepcopy(wpose))

            wpose.position.x += scale * 0.2  # First move up (x)
            wpose.position.z += scale * 0.1  # First move up (z)
            waypoints.append(copy.deepcopy(wpose))
            
  
          #  wpose.position.y -= scale * 0.1  # Third move sideways (y)
           # waypoints.append(copy.deepcopy(wpose))

            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planning, not asking move_group to actually move the robot yet:
            return plan, fraction


     def Give_the_tool(self, scale=1):
            # Copy class variables to local variables to make the web tutorials more clear.
            # In practice, you should use the class variables directly unless you have a good
            # reason not to.
            move_group = self.move_group

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
            #wpose.position.z -= scale * 0.1  # First move up (z)
            #wpose.position.y += scale * 0.2  # and sideways (y)
           # waypoints.append(copy.deepcopy(wpose))

            wpose.position.x -= scale * 0.2  # First move up (x)
            wpose.position.z -= scale * 0.1  # First move up (z)
            waypoints.append(copy.deepcopy(wpose))
          # wpose.position.y -= scale * 0.1  # Third move sideways (y)
           # waypoints.append(copy.deepcopy(wpose))

            # We want the Cartesian path to be interpolated at a resolution of 1 cm
            # which is why we will specify 0.01 as the eef_step in Cartesian
            # translation.  We will disable the jump threshold by setting it to 0.0,
            # ignoring the check for infeasible jumps in joint space, which is sufficient
            # for this tutorial.
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold

            # Note: We are just planning, not asking move_group to actually move the robot yet:
            return plan, fraction
            ## END_SUB_TUTORIAL

     def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        # joint_goal[0] = joint_goal[0] + 3.14/4  # turn
        # joint_goal[1] = 0
        # joint_goal[2] = 0 
        #joint_goal[3] = joint_goal[3] - 3.14/2
        # joint_goal[4] = joint_goal[4] - 3.14/2 # turn the wrist 2
        # joint_goal[5] = joint_goal[5] + 3.14/2  # 1/6 of a turn of the wrist 3

        # joint_goal = [-4.267534081135885, -1.8209956328021448, 2.1080198287963867, -3.2752087751971644, 4.713043689727783, -2.6624208132373255]


        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)



     def go_to_joint_state2(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        # joint_goal[0] = joint_goal[0] + 3.14/4  # turn
        # joint_goal[1] = 0
        # joint_goal[2] = 0 c
        joint_goal[3] = joint_goal[3] + 3.14/2
        # joint_goal[4] = joint_goal[4] - 3.14/2 # turn the wrist 2
        # joint_goal[5] = joint_goal[5] + 3.14/2  # 1/6 of a turn of the wrist 3

        # joint_goal = [-4.267534081135885, -1.8209956328021448, 2.1080198287963867, -3.2752087751971644, 4.713043689727783, -2.6624208132373255]


        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)



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


     def go_to_pose_goal(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(8.0)
        move_group.set_max_velocity_scaling_factor(0.3)
        move_group.set_max_acceleration_scaling_factor(0.2)
        move_group.set_num_planning_attempts(10)
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        # pose of the tape

        pose_goal.orientation.x = -0.9944051959988094
        pose_goal.orientation.y = -0.07356902914468358
        pose_goal.orientation.z = -0.017937575428294494
        pose_goal.orientation.w = 0.07364881200014343
        pose_goal.position.x = -0.4615398406546578
        pose_goal.position.y = 0.5402548784381651
        pose_goal.position.z = 0.05983566848028851
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

        ## END_SUB_TUTORIA

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
    
     def go_to_pose_goal2(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        move_group.set_planner_id('AnytimePathShortening')
        move_group.set_planning_time(8.0)
        move_group.set_max_velocity_scaling_factor(0.3)
        move_group.set_max_acceleration_scaling_factor(0.2)
        move_group.set_num_planning_attempts(10)
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion  for this group to a desired pose for the
        ## end-effector:
        ## pose of the giving
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = -0.29905477161126354
        pose_goal.orientation.y = -0.9536846034637316
        pose_goal.orientation.z = 0.021728881740268994
        pose_goal.orientation.w = 0.02407854628247695
        pose_goal.position.x = -0.6483408942220915
        pose_goal.position.y = -0.38150866299045577
        pose_goal.position.z = 0.1276265145416368


        pose_current = move_group.get_current_pose().pose
        print(pose_current) 
        ## Now, we call the planner to compute the plan and execute it.
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
        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)


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


     def add_box(self, timeout=4):
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
        box_pose.pose.position.z = -0.03      # above the panda_hand frame
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(3, 3, 0.09))

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

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene between the fingers:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.03      # above the panda_hand frame
        box_pose.pose.position.y = -0.25   # above the panda_hand frame
  
        box_name2 = "box2"
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

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
     

def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("Pick and Place pipeline")
        print("----------------------------------------------------------")
        print("Press Ctrl-D to exit at any time")
        print("")
        input("============ Press `Enter` to begin setting up moveitcommander ...")
        input("============ Press `Enter` to plan and display the path ...")

        plantherobot = movetodesiredposition()
        # plantherobot.display_trajectory()
        plantherobot.go_to_pose_goal2()
        input("Press another key to hand it over")
        # plantherobot.go_to_joint_state()
        # plantherobot.go_to_joint_state2()
        # cartesian_plan2, fraction = plantherobot.Give_the_tool()
        # plantherobot.display_trajectory(cartesian_plan2)
        # plantherobot.execute_plan(cartesian_plan2)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
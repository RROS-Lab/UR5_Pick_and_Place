#!/usr/bin/env python3



# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import time
import csv


from std_msgs.msg import Float32MultiArray, Float32
from std_msgs.msg import Bool


try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


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


class UR5MoveGroupPython(object):

    def __init__(self):
        super(UR5MoveGroupPython, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur5_move_group_python", anonymous=True)

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
        group_name = "ur5"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",         
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

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

        temp_sub = rospy.Subscriber('/monitored_points', Float32MultiArray, self.temp_sub_callback)

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.temp = None
        self.gripper_pub = rospy.Publisher('gripper_close', Bool, queue_size=10)
    
    def temp_sub_callback(self,temp_sub):
        self.temp = temp_sub.data
        

    def go_to_joint_state(self,joint_goal):
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
        # joint_goal = move_group.get_current_joint_values()
        # joint_goal[0] = 0
        # joint_goal[1] = -tau / 8
        # joint_goal[2] = 0
        # joint_goal[3] = -tau / 4
        # joint_goal[4] = 0
        # joint_goal[5] = tau / 6  # 1/6 of a turn

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.4
        pose_goal.position.y = 0.1
        pose_goal.position.z = 0.4

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

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
        wpose.position.z -= scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

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

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL

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
    def get_current_pose(self):
        current_pose = self.move_group.get_current_pose().pose
        return current_pose

    def go_to_pose_goal_linear(self, pose_goal,request_input = True):
    # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        # wpose = move_group.get_current_pose().pose
        waypoints = [pose_goal]



        state = self.robot.get_current_state()
        (plan, fraction) = move_group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.01,        # eef_step
                                        0.0)         # jump_threshold
        plan = move_group.retime_trajectory(state, plan, .5)
        
        if request_input:
            self.display_trajectory(plan)
            input("execute?")

        move_group.execute(plan, wait=True)


        return plan
    def csv_to_traj(self, string):
        with open(string) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            lc = 0
            x = []
            y = []
            z = []
            point_num = []
            for row in csv_reader:
                # print(row[0],row[1],row[2],)
                x.append(copy.deepcopy(float(row[0])))
                y.append(copy.deepcopy(float(row[1])))
                z.append(copy.deepcopy(float(row[2])))
                point_num.append(copy.deepcopy(float(row[3])))
                traj = [x, y, z, point_num]
        return traj


    def gripper(self, gripper_close_state):
        self.gripper_pub.publish(gripper_close_state)

    def get_current_joint_values(self):
        return self.move_group.get_current_joint_values()


def main():
    try:
        input(
            "============ Press `Enter` to begin the tutorial by setting up the moveit_commander ..."
        )
        ur_move = UR5MoveGroupPython()
        ur_move.gripper(False)
        # input("============ Press `Enter` to execute a movement using a pose goal ...")
        # ur_move.go_to_pose_goal()
        joint_goal = ur_move.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -55*(pi/180)
        joint_goal[2] = -126*(pi/180)
        joint_goal[3] = -200*(pi/180)
        joint_goal[4] = -90*(pi/180)
        joint_goal[5] = 0
        ur_move.go_to_joint_state(joint_goal)
        input("pick up object?")
        joint_goal[0] = -165*(pi/180)
        joint_goal[1] = -133*(pi/180)
        joint_goal[2] = -47*(pi/180)
        joint_goal[3] = -264*(pi/180)
        joint_goal[4] = -95*(pi/180)
        joint_goal[5] = -65*(pi/180)
        ur_move.go_to_joint_state(joint_goal)

        joint_goal[0] = -162*(pi/180)
        joint_goal[1] = -145*(pi/180)
        joint_goal[2] = -47*(pi/180)
        joint_goal[3] = -256*(pi/180)
        joint_goal[4] = -95*(pi/180)
        joint_goal[5] = -65*(pi/180)
        ur_move.go_to_joint_state(joint_goal)
        
        ur_move.gripper(True)
        time.sleep(1)
        joint_goal[0] = -165*(pi/180)
        joint_goal[1] = -133*(pi/180)
        joint_goal[2] = -47*(pi/180)
        joint_goal[3] = -264*(pi/180)
        joint_goal[4] = -95*(pi/180)
        joint_goal[5] = -65*(pi/180)
        ur_move.go_to_joint_state(joint_goal)

        joint_goal = ur_move.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -55*(pi/180)
        joint_goal[2] = -126*(pi/180)
        joint_goal[3] = -200*(pi/180)
        joint_goal[4] = -90*(pi/180)
        joint_goal[5] = 0
        ur_move.go_to_joint_state(joint_goal)
        input("drop object?")
        ur_move.gripper(False)
        # start_pose = ur_move.get_current_pose()
        # print(start_pose)
        # pose = start_pose
        # input("go to pose?")
        # print(ur_move.temp)
        # traj = ur_move.csv_to_traj('policy0.csv')
        # print(traj)
        # while not rospy.is_shutdown():
        #     # for i in range(len(traj[0])):
                

        #     #     pose.position.x = traj[0][i]
        #     #     pose.position.y = traj[1][i]
        #     #     pose.position.z = traj[2][i]
        #     #     point_num = traj[3][i]
        #     #     if ur_move.temp[int(point_num)]>140:
        #     #         print("skipped "+str(point_num))
        #     #         continue

        #         print("moving to node "+str(point_num))
                
        #     #     ur_move.go_to_pose_goal_linear(pose,False)
        #     #     input()
        #     #     # input()
        #     #     while ur_move.temp[int(point_num)]<140 and not rospy.is_shutdown():
        #     #         print("heating")
        #     #         time.sleep(.1)
        #         if rospy.is_shutdown():
        #             break 


        





        print("============ Python tutorial demo complete!")
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


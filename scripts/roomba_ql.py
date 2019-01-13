#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

class ToiletCleaning:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/odom',Odometry,self.getOdomCB)
        self.scan_sub = rospy.Subscriber('/scan',LaserScan,self.getLaserDistCB)
        self.amcl_sub = rospy.Subscriber('/amcl',PoseWithCOvarianceStamped,getPoseCB)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)

        self.state = 'CHECK_SENSOR'
        self.laser_dists = np.zeros(10)
        self.robot_pose = np.zeros(3) # x y theta
        self.base_pose = Twist()
        self.inversion = 1.0 
        self.reward_grid = np.zeros(15,17)
        self.is_safety = True

    # -- Callback Functions -->
    def getOdomCB(self,msg):
        self.base_pose.linear.x = msg.pose.pose.position.x
        self.base_pose.linear.y = msg.pose.pose.position.y
        euler = tf.transformations.euler_from_quaternion((0,0, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.base_pose.angular.z = euler[2] * self.inversion

    def getLaserDistCB(self,msg):
        for i in range(10):
            self.laser_dists[i] = msg.ranges[i*80]

        # check safety
        for i in range(2):
            if self.laser_dist[i*720] < 0.18:
                self.stopBase()
                self.safety = False
                return 
        for i in range(8):
            if self.laser_dist[80+i*80] < 0.05:
                self.stopBase()
                self.safety = False
                return

    def getPoseCB(self,msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        euler = tf.transformations.euler_from_quaternion((0,0, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.robot_pose[2] = euler[2]

    # -- Control Base Functions -->
    def stopBase(self):
        vel = Twist()
        vel.linear.x = 0
        vel.angular.z = 0
        self.cmd_vel_pub.publish(vel)

    def advanceBase(self,goal_dist):
        if goal_dist < 0:
            print "input positive number!"
            return
        init_x = self.base_pose.linear.x
        init_y = self.base_pose.linear.y
        moved_dist = 0.0
        vel = Twist()
        while moved_dist < goal_dist and not rospy.is_shutdown():
            moved_dist = np.sqrt(pow(init_x - self.base_pose.linear.x,2)
                                +pow(init_y - self.base_pose.linear.y,2))
            print moved_dist
            vel.angular.z = 0.01 # to correct
            vel.linear.x = 0.1
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.1)
        self.stopBase()
        print "finish advancing"

    def rotateBase(self,goal_angle):
        if goal_angle < -1*np.pi or np.pi < goal_angle:
            print "goal angle must be set from -pi to pi."
            return
        init_angle = self.base_pose.angular.z
        moved_angle = 0.0
        vel = Twist()
        if goal_angle > 0:# Counter Clock Wise
            while moved_angle < goal_angle-0.04 and not rospy.is_shutdown():
                moved_angle = self.base_pose.angular.z - init_angle
                if moved_angle <= -np.pi+0.05:
                    print "over horizone"
                    moved_angle = 2*np.pi + moved_angle
                vel.angular.z = 0.1
                self.cmd_vel_pub.publish(vel)
                rospy.sleep(0.01)
                print goal_angle,moved_angle 

        elif goal_angle < 0:# Clock Wise
            while moved_angle > goal_angle+0.04 and not rospy.is_shutdown():
                moved_angle = self.base_pose.angular.z - init_angle
                if moved_angle >= np.pi-0.05:
                    print "over horizone"
                    moved_angle = -2*np.pi + moved_angle
                vel.angular.z = -0.1
                self.cmd_vel_pub.publish(vel)
                rospy.sleep(0.01)
                print goal_angle,moved_angle 
        self.stopBase()
        print "finish roteting"

    # -- Public Functions -->
    def initEnv(self):
        self.reward_grid = np.zeros(15,17)
        self.reward_grid[int(self.robot_pose[0]),int(self.robot_pose[1])] = 1
        self.is_safety = True

    def doAction(self,action):
        #if is_safety == False:
        #    exit()

        if action == 0:
            self.advanceBase(0.1)
        elif action == 1:
            self.rotateBase(np.pi/9) # 20 degree
        elif action == 2:
            self.rotateBase(np.pi/9*-1)

        rospy.sleep(1.0)
        return self.is_safety

    def checkReward():
        if self.reward_grid[int(self.robot_pose[0]),int(self.robot_pose[1])] == 0:
            self.reward_grid[int(self.robot_pose[0]),int(self.robot_pose[1])] == 1
            return 1
        else:
            return 0



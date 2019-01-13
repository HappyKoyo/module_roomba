#!/usr/bin/env python
# -*- coding: utf-8 -*
# general
import numpy as np
import tensorflow as tf
import subprocess
import random
from collections import deque
# ros
import rospy
from std_msgs.msg import Float32,Bool,String
from geometry_msgs.msg import Twist,Pose
from darknet_ros_msgs.msg import BoundingBoxes
from std_srvs.srv import Empty

class Module:
    def __init__(self):
        # subscriber
        self.bb_sub = rospy.Subscriber('/darknet_ros/bounding_boxes',BoundingBoxes,self.getBB,queue_size=1)
        # publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        # service client
        #self.reset_srv  = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # define variable
        self.before_dists = [1.0,1.0]
        self.bb = BoundingBoxes()
        self.target = {"x":0,"y":0,"area":0,"is_detected":False,"xmax":0,"xmin":0,"ymax":0,"ymin":0}
        self.garbage= {"x":0,"y":0,"area":0,"is_detected":False,"xmax":0,"xmin":0,"ymax":0,"ymin":0}

    # reward 
    def getModelCB(self,msg):
        self.model_states = msg

    # vision
    def getBB(self,msg):
        self.bb = msg

    def getObjStates(self):
        self.target["is_detected"] = False
        self.garbage["is_detected"] = False

        for i in range(len(self.bb.bounding_boxes)):
            xmin = float(self.bb.bounding_boxes[i].xmin)
            xmax = float(self.bb.bounding_boxes[i].xmax)
            ymin = float(self.bb.bounding_boxes[i].ymin)
            ymax = float(self.bb.bounding_boxes[i].ymax)
            if self.bb.bounding_boxes[i].Class == "target":
                self.target["x"] = ((xmin+(xmax-xmin)/2))
                self.target["xmax"]=xmax
                self.target["xmin"]=xmin
                self.target["y"] = ((ymin+(ymax-ymin)/2))
                self.target["ymax"]=ymax
                self.target["ymin"]=ymin
                self.target["area"] = (xmax-xmin)*(ymax-ymin)
                self.target["is_detected"] = True
            else:#if self.bb.bounding_boxes[i].Class == "garbage":
                self.garbage["x"] = (xmin+(xmax-xmin))
                self.garbage["xmax"]=xmax
                self.garbage["xmin"]=xmin
                self.garbage["y"] = (ymin+(ymax-ymin))
                self.garbage["ymax"]=ymax
                self.garbage["ymin"]=ymin                
                self.garbage["area"] = (xmax-xmin)*(ymax-ymin)
                self.garbage["is_detected"] = True

        #print self.garbage["area"],self.target["area"]
        return self.target["x"] ,self.target["area"] ,float(self.target["is_detected"]),\
               self.garbage["x"],self.garbage["area"],float(self.garbage["is_detected"])

    # move
    def moveRoomba(self,x,yaw):
        vel = Twist()
        vel.linear.x  = x
        vel.angular.z = yaw
        self.cmd_vel_pub.publish(vel)


    # loop
    def main_loop(self):
        rospy.sleep(3)
        sleep_rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            #added by Enomoto
            sleep_rate.sleep()
            self.getObjStates()
            if self.garbage["xmin"] >= self.target["xmax"]:
                Distx=self.garbage["x"]-self.target["x"]
                Disty=self.target["y"]-self.garbage["y"]
                print "dist x is ",Distx
                
                self.moveRoomba(0,2)
		rospy.sleep(0.5)
                self.moveRoomba(0,2)
		rospy.sleep(0.5)
                self.moveRoomba(0,2)
		rospy.sleep(0.5)

                self.moveRoomba(0.1,0)
       		rospy.sleep(0.5)
                self.moveRoomba(0.1,0)
       		rospy.sleep(0.5)

                self.moveRoomba(0,-2)
		rospy.sleep(0.5)
                self.moveRoomba(0,-2)
		rospy.sleep(0.5)
                self.moveRoomba(0,-2)
		rospy.sleep(0.5)
                self.moveRoomba(0,-0.2)
		rospy.sleep(0.5)

                
            elif self.garbage["xmax"] <= self.target["xmin"]:
                Distx=self.target["x"]-self.garbage["x"]
                Disty=self.target["y"]-self.garbage["y"]
                
                self.moveRoomba(0,-2)
		rospy.sleep(0.5)
                self.moveRoomba(0,-2)
		rospy.sleep(0.5)
                self.moveRoomba(0,-2)
		rospy.sleep(0.5)

                self.moveRoomba(0.1,0)
       		rospy.sleep(0.5)
                self.moveRoomba(0.1,0)
       		rospy.sleep(0.5)

                self.moveRoomba(0,2)
		rospy.sleep(0.5)
                self.moveRoomba(0,2)
		rospy.sleep(0.5)
                self.moveRoomba(0,2)
		rospy.sleep(0.5)
                self.moveRoomba(0,0.2)
		rospy.sleep(0.5)
   
            elif self.garbage["xmax"]-self.garbage["xmin"] >= 230:
                print self.garbage["xmax"]-self.garbage["xmin"]
                break

            else:
                self.moveRoomba(0.1,0)
                         

if __name__ == '__main__':
    rospy.init_node('module')
    module = Module()
    module.main_loop()


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
from gazebo_msgs.msg import ModelStates
from darknet_ros_msgs.msg import BoundingBoxes
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

NUM_EPISODES = 299  # number of episodes
MAX_STEPS = 50 # number of steps each episode
GOAL_MEAN_REWARD = 195
CONSECUTIVE_ITERATIONS = 10  # number of trials to calculate the mean value
# ---
GAMMA = 0.99            # discount factor
HIDDEN_SIZE = 16        # number of hidden layer neuron 
LEARNING_RATE = 0.01 #0.00001 # learning rate of q-network
MEMORY_SIZE = 10000     # buffer memory size
BATCH_SIZE = 32         

def huberloss(y_true, y_pred):
    print y_true,y_pred
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=6, action_size=2, hidden_size=20):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
 
    #Experience Replay
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 6))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)
 
        for i, (state_b, advance, roll, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            #target = reward_b
            target_advance = reward_b
            target_roll    = reward_b
 
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b)[0]
                # retmain[x] x->0:advance 1:role
                #next_advance = retmainQs[0]
                #next_roll    = retmainQs[1]
                #target_advance = reward_b + gamma * targetQN.model.predict(next_state_b)[0][0]
                #target_advance = reward_b*advance + gamma * targetQN.model.predict(next_state_b)[0][0]
                #target_role    = reward_b*roll    + gamma * targetQN.model.predict(next_state_b)[0][1]
                target_advance = reward_b*advance
                target_role    = reward_b*roll
                #print advance,reward_b,target_advance
                
                #target_advance = 10#reward_b*advance
                #target_role    = -10#reward_b*roll
                
            targets[i] = self.model.predict(state_b)    # output of q-network
            targets[i][0] = target_advance
            targets[i][1] = target_role
            print reward_b,targets[i][1],roll
            #targets[i][1] = roll                
        #print inputs

        #print targets
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochs is iteration of data 

# Experience Memory
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, experience):
        self.buffer.append(experience)
 
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
 
    def len(self):
        return len(self.buffer)

    def reset(self):#added
        self.buffer = deque(maxlen=MEMORY_SIZE)

# select action
class Actor:
    def get_action(self, state, episode, targetQN):   # [C]ｔ＋１での行動を返す
        # ε-greedy
        epsilon = 0.001 + 0.9 / (1.0+episode)
        #epsilon = 0.0001# + 0.9 / (2.0+episode)
        #epsilon = 1 - 0.05*episode
        #epsilon = 0.3
 
        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = targetQN.model.predict(state)[0]
            advance = retTargetQs[0]
            roll    = retTargetQs[1]
            #print state,retTargetQs
            #print retTargetQs
            #print "net roll =",roll,"advance",advance
 
        else:
            advance = (random.random()-0.5)*2
            roll    = (random.random()-0.5)*2
            #print "rnd roll =",roll,"advance",advance

        return advance,roll

class Module:
    def __init__(self):
        # subscriber
        self.model_sub  = rospy.Subscriber('/gazebo/model_states',ModelStates,self.getModelCB,queue_size=1)
        self.bb_sub = rospy.Subscriber('/darknet_ros/bounding_boxes',BoundingBoxes,self.getBB,queue_size=1)
        # publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        # service client
        self.set_obj_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        #self.reset_srv  = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # define variable
        self.before_dists = [1.0,1.0]
        self.bb = BoundingBoxes()
        self.target = {"x":0,"y":0,"area":0,"is_detected":False}
        self.garbage= {"x":0,"y":0,"area":0,"is_detected":False}

    # reward 
    def getModelCB(self,msg):
        self.model_states = msg

    def getReward(self,advance,roll):
        obj=self.model_states
        dist_x = abs(obj.pose[1].position.x - obj.pose[2].position.x)
        dist_y = abs(obj.pose[1].position.y - obj.pose[2].position.y)

        reward = 0
        #old reward
        if dist_x < 0.5 and dist_y < 0.5:
            reward += 10
        else:
            reward += (self.before_dists[0]+self.before_dists[1])-(dist_x+dist_y)

        if self.target["is_detected"] == False or self.garbage["is_detected"] == False:
            reward += -10
            print "can't recognize"
        self.before_dists = [dist_x,dist_y]


        # only roll
        if self.target["x"] > 0 and roll > 0:
            reward += -1
        elif self.target["x"] > 0 and roll < 0:
            reward += 1
        elif self.target["x"] < 0 and roll < 0:
            reward += -1
        elif self.target["x"] < 0 and roll > 0:
            reward += 1
        #elif self.target["x"] < -0.2 and 0.2 < self.target["x"]:
        #    reward = 1

        if advance <= 0:
            reward -= 1
        else:
            reward += 1

        #reward = 0
        #if self.target["x"] >= 0.2 and roll >= 0.2:
        #    reward = -1
        #elif self.target["x"] >= 0.2 and roll < 0.2:
        #    reward = 1
        #elif self.target["x"] <= -0.2 and roll <= -0.2:
        #    reward = -1
        #elif self.target["x"] <= -0.2 and roll > -0.2:
        #    reward = 1
        #elif self.target["x"] < -0.2 and 0.2 < self.target["x"]:
        #    reward = -1

        #if advance <= 0:
        #    reward += 10

        print "reward is",reward
        #print self.target["x"]
        return reward

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
                self.target["x"] = ((xmin+(xmax-xmin)/2)-320)/320.0
                self.target["y"] = ((ymin+(ymax-ymin)/2)-240)/240.0
                self.target["area"] = (xmax-xmin)*(ymax-ymin)/307200.0
                self.target["is_detected"] = True
            else:#if self.bb.bounding_boxes[i].Class == "garbage":
                self.garbage["x"] = (xmin+(xmax-xmin)/2-320)/320.0
                self.garbage["y"] = (ymin+(ymax-ymin)/2-240)/240.0
                self.garbage["area"] = (xmax-xmin)*(ymax-ymin)/307200.0
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

    # simulater controll
    def setObject(self,x,y,object_name):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        twist = Twist()
        model_state = SetModelState
        model_state.model_name = object_name
        model_state.pose = pose
        model_state.twist = twist
        model_state.reference_frame = "world"
        self.set_obj_srv(model_state)

    def resetWorld(self):
        self.setObject(0,0,"create_2")
        y = random.random()
        #self.setObject(0.5,(y-0.5)/2,"coke0")
        self.setObject(0.5,(y-0.5)/2,"coke0")
        y = random.random()
        self.setObject(1.5,y-0.5,"beer0")

    # loop
    def main_loop(self):
        rospy.sleep(3)
        islearned = False
        mainQN = QNetwork(hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE)     # main q-network
        targetQN = QNetwork(hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE)   # target q-network
        #mainQN.model.load_weights('module_weights_new.h5')
        memory = Memory(max_size=MEMORY_SIZE)
        actor = Actor()
        self.resetWorld()
        tx,ta,td,gx,ga,gd = self.getObjStates()
        
        state = np.array([tx,ta,td,gx,ga,gd])
        state = np.reshape(state, [1, 6])
        for episode in range(NUM_EPISODES):
            episode_reward = 0
            targetQN.model.set_weights(mainQN.model.get_weights())
            if episode % 10 == 9:
                targetQN.model.save_weights('/home/demulab/catkin_ws/src/module/weights/module_weights'+str(episode)+'.h5')
            self.resetWorld()
            for t in range(MAX_STEPS + 1):
                r = rospy.Rate(5)
                r.sleep()
                advance,roll= actor.get_action(state, episode, mainQN)   # select action 
                #print advance
                #next_state, reward, done, info = env.step(action)   # calculate s_{t+1},_R{t}
                #self.moveRoomba(advance*0.6,roll)
                self.moveRoomba(advance,roll)
                tx,ta,td,gx,ga,gd = self.getObjStates()
                #print tx,ta,td,gx,ga,gd
                #next_state = np.array([tx,ta,td,gx,ga,gd])
                next_state = np.array([tx,ta,td,gx,ga,gd])
                next_state = np.reshape(next_state, [1, 6])
                reward = self.getReward(advance,roll)
                episode_reward += reward
                memory.add((state, advance, roll, reward, next_state)) # save this experience to memory
                if (memory.len() > BATCH_SIZE) and not islearned:
                    mainQN.replay(memory, BATCH_SIZE, GAMMA, targetQN)
                    memory.reset()

                state = next_state
		targetQN.model.set_weights(mainQN.model.get_weights())

                # game finished
                if reward >= 7 or self.target["is_detected"] == False or self.garbage["is_detected"] == False:
                    break
            print "episode_reward is",episode_reward

if __name__ == '__main__':
    rospy.init_node('module')
    module = Module()
    module.main_loop()

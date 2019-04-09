import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

from random import randint
import collections
import numpy as np

from DQNAgent import DQNAgent

action_map = {}
action_arr = [3, 3, 2]
index = 0

for depth1 in range(action_arr[0]):
    for depth2 in range(action_arr[1]):
        for depth3 in range(action_arr[2]):
            action_map[index] = []
            action_map[index].append(depth1)
            action_map[index].append(depth2)
            action_map[index].append(depth3)
            index += 1

class RLActorAPI(TFPluginAPI):

    # expected optional api: setup your model for training
    def onSetup(self):
        ue.log('On Setup!')
        self.num_actions =18
        null_input = np.zeros(3)
        self.state_size = null_input.shape[0]
        self.model = DQNAgent(state_size=self.state_size, action_size=self.num_actions)

        self.scores = []
        self.episodes = []

        self.state = np.reshape(null_input, [1, self.state_size])
        self.action = 0

    # expected optional api: parse input object and return a result object, which will be converted to json for UE4
    def onJsonInput(self, jsonInput):
        # LSM BEGIN
        action = self.model.get_action(self.state)
        self.model.append_sample(state=self.state, action=action, reward=0, next_state=self.state, done=False)
        '''
        ballPos = jsonInput['ballPosition']
        next_state = [jsonInput['paddlePosition'], ballPos['x'], ballPos['y']]
        reward = jsonInput['actionScore']
        done = jsonInput['done']
        
        next_state = np.reshape(next_state, [1, self.state_size])

        self.model.append_sample(state=self.state, action=action, reward=reward, next_state=next_state, done=done)
        self.state = next_state

        if len(self.model.memory) >= self.model.train_start:
            self.model.train_model()

        if done:
            self.model.update_target_model()
        '''
        return {
            'MoveForward': action_map[action][0],
            'MoveRight' : action_map[action][1],
            'Attack' : action_map[action][2]
        }

    # expected optional api: start training your network
    def onBeginTraining(self):
        pass


# required function to get our api
def getApi():
    return RLActorAPI.getInstance()
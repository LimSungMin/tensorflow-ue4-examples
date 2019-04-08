import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

# utility imports
from random import randint
import collections
import numpy as np

from DQNAgent import DQNAgent

class ExampleAPI(TFPluginAPI):

    # expected optional api: setup your model for training
    def onSetup(self):
        self.num_actions = 3
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

        return {'action': float(action)}

    # custom function to determine which paddle we are
    def setPaddleType(self, type):
        self.paddle = 0
        if (type == 'PaddleRight'):
            self.paddle = 1


    # expected optional api: start training your network
    def onBeginTraining(self):
        pass


# NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
# required function to get our api
def getApi():
    # return CLASSNAME.getInstance()
    return ExampleAPI.getInstance()
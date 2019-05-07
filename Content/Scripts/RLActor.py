import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

from random import randint
import collections
import numpy as np

from DQNAgent import DQNAgent

action_map = {}
action_arr = [3, 3, 2] # MoveForward, MoveRight, Attack
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
        # 행동 수 정의
        self.num_actions =18
        state_input = np.zeros(3)
        self.state_size = state_input.shape[0]
        # DQN 모델 생성
        self.model = DQNAgent(state_size=self.state_size, action_size=self.num_actions)
        self.scores = []
        self.episodes = []

        self.state = np.reshape(state_input, [1, self.state_size])
        self.action = 0

    # expected optional api: parse input object and return a result object, which will be converted to json for UE4
    def onJsonInput(self, jsonInput):
        # state 에 대한 action 을 가져옴
        action = self.model.get_action(self.state)
        # 다음 스테이트
        next_state = [
            jsonInput['distance_to_target'],
            jsonInput['my_health'],
            jsonInput['time_alive']
        ]
        reward = jsonInput['reward']
        done = jsonInput['done']

        next_state = np.reshape(next_state, [1, self.state_size])
        # DQN replay memory 에 샘플링
        self.model.append_sample(state=self.state, action=action, reward=reward, next_state=next_state, done=done)
        self.state = next_state
        # 초기에 일정 랜덤 액션 후 학습 시작
        if len(self.model.memory) >= self.model.train_start:
            self.model.train_model()
        # 게임이 완료되면 타겟 모델을 업데이트 함
        if done:
            ue.log('Update Target Model')
            self.model.update_target_model()
            self.ReportGame(jsonInput['time_alive'], jsonInput['valid_attack_count'])
        # 다음에 수행할 action 을 return
        return {
            'MoveForward': action_map[action][0],
            'MoveRight' : action_map[action][1],
            'Attack' : action_map[action][2]
        }

    # expected optional api: start training your network
    def onBeginTraining(self):
        pass

    def ClearSession(self, jsonInput):
        ue.log('Clear Session Called')
        tf.keras.backend.clear_session()
        # self.model.clear_session()

    def ReportGame(self, alive_time, valid_attack_count):
        f= open('C:/Users/lsm_o/Documents/git/tensorflow-ue4-examples/Content/Scripts/log/report_game.log', 'a')
        f.write(str(valid_attack_count) + "\n")
        f.close()


# required function to get our api
def getApi():
    return RLActorAPI.getInstance()
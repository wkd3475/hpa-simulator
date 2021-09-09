import math
import numpy as np
import random
from autoscaler.rl import *

class HPA_Q_Learning_Trend(RL):
    def __init__(self, rl_config):
        self.pods_min = rl_config.pods_min
        self.pods_max = rl_config.pods_max
        self.a_history = []
        self.s_history = []
        self.r_history = []
        # (utilization, # of pods, actions)
        self.Q = np.zeros([11, self.pods_max-self.pods_min+2, 11, self.pods_max-self.pods_min+2])
        self.action_space = list(range(self.pods_min, self.pods_max+1))

        self.alpha = rl_config.alpha
        self.gamma = rl_config.gamma
        self.epsilon = rl_config.epsilon
        # m4.xlarge => 4 vCPU => 0.2 USD / hour
        # 1 vCPU => 0.05 USD / hour
        # pod => 0.2 core => 0.01 USD
        # error => 0.0005 USD
        self.resource_cost = rl_config.resource_cost
        self.violation_cost = rl_config.violation_cost
        self.autoscaling_period = rl_config.autoscaling_period

    def monitoring(self, obs):
        state = {}
        reward = None

        u = int(float(obs.U) // 0.1)
        c = int(obs.C[-1])
        c_avg = sum(obs.C) / len(obs.C)
        e = sum(obs.E)
        x = np.arange(0, len(obs.A))
        y = np.array(obs.A)
        z = np.polyfit(x,y,1)
        trend = z[0]//1
        if trend > 5:
            trend = 5
        if trend < -5:
            trend = -5

        reward = -1 * self.resource_cost * c_avg * self.autoscaling_period + -1 * self.violation_cost * e
        
        state['u'] = u
        state['c'] = c-self.pods_min
        state['t'] = int(trend+5)

        return state, reward

    def action_to_desired(self, action):
        return self.action_space[action]

    def get_action(self, state):
        max_q = float('-inf')
        max_a = []
        if np.random.rand() < self.epsilon:
            return random.choice(list(range(len(self.action_space))))

        for i in range(len(self.action_space)):
            if max_q < self.Q[state['u'], state['c'], state['t'], i]:
                max_q = self.Q[state['u'], state['c'], state['t'], i]
                max_a = [i]
            elif max_q == self.Q[state['u'], state['c'], state['t'], i]:
                max_a.append(i)
        
        action = random.choice(max_a)

        self.a_history.append(action)
        return action

    def update(self, s, a, r, s_next):
        self.Q[s['u'], s['c'], a] = self.Q[s['u'], s['c'], s['t'], a] + self.alpha * (r + self.gamma * np.nanmax(self.Q[s_next['u'], s_next['c'], s_next['t'],: ]) - self.Q[s['u'], s['c'], s['t'], a])

    def get_action_with_noisy(self, state):
        max_q = float('-inf')
        max_actions = []

        for i in range(len(self.action_space)):
            if max_q < self.Q[state['u'], state['c'], state['t'], i]:
                max_q = self.Q[state['u'], state['c'], state['t'], i]
                max_actions = [i]
            elif max_q == self.Q[state['u'], state['c'], state['t'], i]:
                max_actions.append(i)
        
        action = random.choice(max_actions)

        if np.random.rand() < self.epsilon:
            noisy_actions = None
            if action == 0:
                noisy_actions = [action, action+1]
            elif action == len(self.action_space)-1:
                noisy_actions = [action-1, action]
            else:
                noisy_actions = [action-1, action, action+1]

            action = random.choice(noisy_actions)

        self.a_history.append(action)
        return action
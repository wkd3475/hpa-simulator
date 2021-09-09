import math
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 96)
        self.fc4 = nn.Linear(96, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.linear(self.fc4(x))
        return x

class RlConfig:
    def __init__(self, 
        pods_min, 
        pods_max, 
        resource_cost, 
        violation_cost, 
        autoscaling_period, 
        learning_rate, 
        discount_factor, 
        epsilon):

        self.pods_min = pods_min
        self.pods_max = pods_max
        self.resource_cost = resource_cost
        self.violation_cost = violation_cost
        self.autoscaling_period = autoscaling_period
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

class HPA_Q_Learning:
    def __init__(self, rl_config):
        self.pods_min = rl_config.pods_min
        self.pods_max = rl_config.pods_max
        self.a_history = []
        self.s_history = []
        self.r_history = []
        # (utilization, # of pods, actions)
        #(2,10)
        self.Q = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(Q.parameters(), lr=0.01)
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

    def convert_obs(self, obs):
        u = int(float(obs.U) // 0.1)
        c = int(obs.C[-1])
        c_avg = sum(obs.C) / len(obs.C)
        e = sum(obs.E)
        reward = -1 * self.resource_cost * c_avg * self.autoscaling_period + -1 * self.violation_cost * e
        state = (u, c)
        self.s_history.append(state)
        self.r_history.append(reward)
        return state, reward
    
    def epsilon_decay(self):
        self.epsilon = self.epsilon * 0.9

    def get_action(self, state):
        max_q = float('-inf')
        max_a = []
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)

        for i in range(self.pods_min, self.pods_max+1):
            if max_q < self.Q[state[0], state[1], i]:
                max_q = self.Q[state[0], state[1], i]
                max_a = [i]
            elif max_q == self.Q[state[0], state[1], i]:
                max_a.append(i)
        
        desired_c = random.choice(max_a)

        self.a_history.append(desired_c)
        return desired_c

    def update(self, s, a, s_next, r_next):
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.alpha * (r_next + self.gamma * np.nanmax(self.Q[s_next[0], s_next[1],: ]) - self.Q[s[0], s[1], a])


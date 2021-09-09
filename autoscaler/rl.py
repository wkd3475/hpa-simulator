from abc import *

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

class RL(metaclass=ABCMeta):
    @abstractmethod
    def monitoring(self):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def action_to_desired(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def epsilon_decay(self):
        e = self.epsilon * 0.9
        if e < 0.1:
            self.epsilon = 0.1
        else:
            self.epsilon = e
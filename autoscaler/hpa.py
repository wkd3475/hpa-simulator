import math

class HpaConfig:
    def __init__(self, pods_min, pods_max, target_utilization, scaling_tolerance):
        self.pods_min = pods_min
        self.pods_max = pods_max
        self.target_utilization = target_utilization
        self.scaling_tolerance = scaling_tolerance

class HPA:
    def __init__(self, hpa_config):
        self.pods_min = hpa_config.pods_min
        self.pods_max = hpa_config.pods_max
        self.target_utilization = hpa_config.target_utilization
        self.scaling_tolerance = hpa_config.scaling_tolerance

    def get_action(self, state):
        u = float(state.U)
        c = float(state.C[-1])

        if (abs(u/self.target_utilization - 1) <= self.scaling_tolerance):
            return int(c)

        desired_c = math.ceil(c * u / self.target_utilization)

        if desired_c > self.pods_max:
            desired_c = self.pods_max
        elif desired_c < self.pods_min:
            desired_c = self.pods_min

        return int(desired_c)
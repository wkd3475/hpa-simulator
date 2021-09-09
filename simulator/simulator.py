import math
import csv
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

@dataclass
class State:
    C: List[int] = field(default_factory=list) # number of pods
    U: float = None # Approximated average CPU Utilization
    A: List[float] = field(default_factory=list)
    E: List[float] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    W: List[float] = field(default_factory=list)

class EnvConfig:
    def __init__(self, 
        application_profile, 
        traffic_history, 
        init_pods=1,
        timeout=10, 
        autoscale_period=60, 
        simulation_period=0.1, 
        readiness_probe=3,
        resource_cost=0.99,
        violation_cost=0.01):

        self.application_profile = application_profile
        self.traffic_history = traffic_history
        self.init_pods = init_pods
        self.timeout = timeout
        self.autoscale_period = autoscale_period
        self.simulation_period = simulation_period
        self.readiness_probe = readiness_probe
        self.resource_cost = resource_cost
        self.violation_cost = violation_cost

class Environment:
    def __init__(self, env_config):
        self.t = 1
        self.last_period = len(env_config.traffic_history)

        self.simulation_period = env_config.simulation_period # seconds
        self.autoscale_period = env_config.autoscale_period # seconds
        self.timeout = env_config.timeout # seconds
        self.readiness_probe = env_config.readiness_probe # seconds

        self.alpha = int(env_config.autoscale_period / env_config.simulation_period)
        self.beta = int(env_config.timeout / env_config.simulation_period)

        self.init_pods = env_config.init_pods

        self.desired_c = env_config.init_pods
        self.c_buffer = [env_config.init_pods] * int(env_config.readiness_probe / env_config.simulation_period) # for readiness_probe buffering
        self.c = [env_config.init_pods]
        self.a = [0]
        self.e = [0]
        self.s = [0]
        self.w = [0]
        
        self.state = [State(C=[self.get_current_c()], U=0, A=[0], E=[0], S=[0], W=[0])]

        self.ap = env_config.application_profile
        # self.plot_history(env_config.traffic_history)
        #histroy의 원소는 1초 동안 requests/s가 몇이었는지를 저장하고 있는다.
        self.traffic_history = self.convert_traffic_history(env_config.traffic_history)
        self.simulation_total_len = len(self.traffic_history)

        self.resource_cost = env_config.resource_cost
        self.violation_cost = env_config.violation_cost

    def get_setting(self):
        setting = {}
        setting["init_pods"] = self.init_pods
        setting["timeout"] = self.timeout
        setting["autoscale_period"] = self.autoscale_period
        setting["simulation_period"] = self.simulation_period
        setting["readiness_probe"] = self.readiness_probe
        return setting

    def plot_history(self, traffic_history):
        result = [0]
        t = [0]
        a = 1
        for h in traffic_history:
            for i in range(5):
                t.append(a)
                result.append(h)
                a = a+1

        plt.figure(figsize=(8,4))
        plt.rc('font', size=11)
        plt.scatter(t[:10001], result[:10001], 2, label='train')
        plt.scatter(t[10001:], result[10001:], 2, label='test')
        # plt.plot(t, result, 'b.', label='NASA-HTTP[20]')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Arrived requests rate (requests/s)')
        plt.legend(loc='upper left')
        plt.show()

    # traffic_history는 5초 마다 프로메테우스가 수집한 테이터를 바탕으로 파악할 수 있음
    # 즉, 하나의 값이 5초 동안의 평균 request rate를 의미함
    def convert_traffic_history(self, traffic_history):
        converted_history = [0]
        
        for h in traffic_history:
            for i in range(int(5/self.simulation_period)):
                converted_history.append(h*self.simulation_period)
                
        return converted_history

    def get_current_state(self):
        state = self.state[-1]
        return state

    def get_current_c(self):
        current_c = self.c[-1]
        return current_c

    def get_x_state(self, x):
        return x[self.t-self.alpha:self.t]

    def get_utilization(self):
        served = sum(self.get_x_state(self.s))
        under = 0
        for i in range(self.t-self.alpha, self.t):
            under = under + (self.c[i] * self.ap(self.c[i]))

        under = under * self.simulation_period
        return served / under

    def scale_to(self, desired_c):
        self.desired_c = desired_c

    def scale(self, action):
        self.desired_c = self.desired_c + action

    # rl state one step
    def next_state(self):
        if self.t + self.alpha >= self.simulation_total_len:
            # print("no more next state")
            return False

        for i in range(self.alpha):
            self.next_simulation_step()

        # 현재 상태
        C = self.get_x_state(self.c)
        U = self.get_utilization()
        A = self.get_x_state(self.a)
        E = self.get_x_state(self.e)
        S = self.get_x_state(self.s)
        W = self.get_x_state(self.w)

        state = State(C, U, A, E, S, W)
        self.state.append(state)
        return state
    
    # simulation one step
    def next_simulation_step(self):
        t = self.t
        beta = self.beta

        c_t = self.c_buffer.pop(0)
        a_t = None
        e_t = None
        s_t = None
        w_t = None
        
        a_t = self.traffic_history[t]

        if t - self.beta < 0:
            e_t = 0
        else:
            sum_of_s = 0
            sum_of_e = 0
            for i in range(1, beta+1):
                sum_of_s = sum_of_s + self.s[t-i]

            for i in range(1, beta):
                sum_of_e = sum_of_e + self.e[t-i]

            result = self.w[t-beta] - sum_of_s - sum_of_e

            if result < 0:
                e_t = 0
            else:
                e_t = result

        w_t = self.w[t-1] - self.s[t-1] + a_t - e_t
        s_t = min(c_t * self.ap(c_t) * self.simulation_period, w_t)

        self.w.append(w_t)
        self.s.append(s_t)
        self.a.append(a_t)
        self.e.append(e_t)
        self.c.append(c_t)
        self.c_buffer.append(self.desired_c)

        self.t = t+1

    # print simulation results
    # def print_result(self):
    #     for i in range(self.t):
    #         print("iter : ", i)
    #         print("a", format(self.a[i], ".2f"))
    #         print("e", format(self.e[i], ".2f"))
    #         print("s", format(self.s[i], ".2f"))
    #         print("w", format(self.w[i], ".2f"))
    #         print("pods", self.c[i])
    #         print("-------------------")

    # save the history of simulation
    def save_result(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            wr = csv.writer(csvfile)
            iterations = []
            for i in range(self.t):
                iterations.append(i)

            wr.writerow(["period"]+iterations)
            wr.writerow(["W"]+self.w)
            wr.writerow(["S"]+self.s)
            wr.writerow(["A"]+self.a)
            wr.writerow(["E"]+self.e)
            wr.writerow(["pods"]+self.c)

    def plot_result(self):
        for a in self.w:
            if a < 0:
                print("warning")
        it = []
        C = []
        U = []
        E = []
        W = []
        A = []
        S = []

        j = 0
        c = 0
        for i in self.state:
            # c = c + i.C
            # C.append(i.C)
            # U.append(i.U)
            A.append(sum(i.A)/self.autoscale_period)
            E.append(sum(i.E)/self.autoscale_period)
            W.append(sum(i.W)/self.autoscale_period)
            S.append(sum(i.S)/self.autoscale_period)
            U.append(i.U)
            C.append(i.C[-1])
            it.append(j)
            j = j+1

        sum_A = sum(A)
        sum_E = sum(E)
        error = sum_E / sum_A * 100
        sum_C = sum(C)
        resource = sum_C / len(C)

        avg_cost = self.resource_cost * resource * len(C) * self.autoscale_period + self.violation_cost * sum_E
        print(f"error result : {error}")
        print(f"average resource : {resource}")
        print(f"average cost : {avg_cost}")
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        # ax3 = fig.add_subplot(2, 1, 2)

        ax1.plot(it, A, 'r', label='A')
        ax1.plot(it, E, 'b', label='E')
        ax2.plot(it, C, 'g', label='C')
        # ax3.plot(it, U, 'r', label='U')

        # plt.plot(it, A, 'r', label='A')
        # plt.plot(it, E, 'b', label='E')
        # plt.plot(it, S, 'g', label='S')
        # plt.xlabel('iterations')
        # plt.ylabel('value')
        # plt.legend(loc='upper left')
        plt.show()
        return avg_cost

    def print_result(self):
        for a in self.w:
            if a < 0:
                print("warning")
        it = []
        C = []
        U = []
        E = []
        W = []
        A = []
        S = []

        j = 0
        c = 0
        for i in self.state:
            # c = c + i.C
            # C.append(i.C)
            # U.append(i.U)
            A.append(sum(i.A)/self.autoscale_period)
            E.append(sum(i.E)/self.autoscale_period)
            W.append(sum(i.W)/self.autoscale_period)
            S.append(sum(i.S)/self.autoscale_period)
            U.append(i.U)
            C.append(i.C[-1])
            it.append(j)
            j = j+1

        sum_A = sum(A)
        sum_E = sum(E)
        error = sum_E / sum_A * 100
        sum_C = sum(C)
        resource = sum_C / len(C)
        sum_U = sum(U)
        avg_util = sum_U / len(U)

        avg_cost = self.resource_cost * resource * len(C) * self.autoscale_period + self.violation_cost * sum_E
        print(f"error result : {error}")
        print(f"average resource : {resource}")
        print(f"average cost : {avg_cost}")
        print(f"average utilization: {avg_util}")
        return avg_cost

    def get_result(self):
        for a in self.w:
            if a < 0:
                print("warning")
        it = []
        C = []
        U = []
        E = []
        W = []
        A = []
        S = []

        j = 0
        c = 0
        for i in self.state:
            A.append(sum(i.A)/self.autoscale_period)
            E.append(sum(i.E)/self.autoscale_period)
            W.append(sum(i.W)/self.autoscale_period)
            S.append(sum(i.S)/self.autoscale_period)
            U.append(i.U)
            C.append(i.C[-1])
            it.append(j)
            j = j+1

        sum_A = sum(A)
        sum_E = sum(E)
        error = sum_E / sum_A * 100
        sum_C = sum(C)
        resource = sum_C / len(C)
        sum_U = sum(U)
        avg_util = sum_U / len(U)

        avg_cost = self.resource_cost * resource * len(C) * self.autoscale_period + self.violation_cost * sum_E
        # print(f"error result : {error}")
        # print(f"average resource : {resource}")
        # print(f"average cost : {avg_cost}")
        return error, resource, avg_cost, avg_util

    def get_everything(self):
        for a in self.w:
            if a < 0:
                print("warning")
        it = []
        C = []
        U = []
        E = []
        W = []
        A = []
        S = []

        j = 0
        c = 0
        for i in self.state:
            # c = c + i.C
            # C.append(i.C)
            # U.append(i.U)
            A.append(sum(i.A)/self.autoscale_period)
            E.append(sum(i.E)/self.autoscale_period)
            W.append(sum(i.W)/self.autoscale_period)
            S.append(sum(i.S)/self.autoscale_period)
            U.append(i.U)
            C.append(i.C[-1])
            it.append(j)
            j = j+1

        sum_A = sum(A)
        sum_E = sum(E)
        error = sum_E / sum_A * 100
        sum_C = sum(C)
        resource = sum_C / len(C)
        sum_U = sum(U)
        avg_util = sum_U / len(U)

        avg_cost = self.resource_cost * resource * len(C) * self.autoscale_period + self.violation_cost * sum_E
        # print(f"error result : {error}")
        # print(f"average resource : {resource}")
        # print(f"average cost : {avg_cost}")
        return error, resource, avg_cost, avg_util, it, A, E, C

    def analyze(self):
        #TODO
        #분석해서 차트 출력
        return 0
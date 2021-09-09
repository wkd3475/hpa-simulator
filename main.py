from simulator.simulator import *
from autoscaler import *
import csv
import datetime
import math
import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit, QPushButton)

def application_profile(num_of_pods):
    return 66.5 + 52.2/num_of_pods

def run_HPA(test_env_config, hpa_config):
    scaler = HPA(hpa_config)
    
    env = Environment(test_env_config)
   
    for i in range(100000):
        state = env.next_state()
        if state == False:
            print("max iter : " + str(i))
            break
        action = scaler.get_action(state)
        env.scale_to(action)
        
    # simulator.print_result()
    # env.save_result('./result/advanced.csv')
    env.print_result()

def offline_q_learning(scaler, env_config, rl, mode):
    env = Environment(env_config)

    s = None
    a = None
    r_prev = None
    s_prev = None
    a_prev = None

    for i in range(100000):
        obs = env.next_state()
        if obs == False:
            # print("max iter : " + str(i))
            break

        s, r_prev = scaler.monitoring(obs)
        a = scaler.get_action(s)
        if rl == "q_learning":
            env.scale_to(scaler.action_to_desired(a))
        elif rl == "q_leanring_fixed_action":
            env.scale(scaler.action_to_desired(a))
        elif rl == "q_learning_trend":
            env.scale_to(scaler.action_to_desired(a))

        if s_prev is not None:
            if mode == "train":
                scaler.update(s_prev, a_prev, r_prev, s)

        s_prev = s
        a_prev = a

    return env

def online_q_learning(scaler, env_config, rl):
    scaler.epsilon = 1.0
    env = Environment(env_config)

    s = None
    a = None
    r_prev = None
    s_prev = None
    a_prev = None

    for i in range(100000):
        obs = env.next_state()
        if obs == False:
            # print("max iter : " + str(i))
            break

        s, r_prev = scaler.monitoring(obs)
        a = scaler.get_action_with_noisy(s)
        if rl == "q_learning":
            env.scale_to(scaler.action_to_desired(a))
        elif rl == "q_leanring_fixed_action":
            env.scale(scaler.action_to_desired(a))
        elif rl == "q_learning_trend":
            env.scale_to(scaler.action_to_desired(a))

        if s_prev is not None:
            scaler.update(s_prev, a_prev, r_prev, s)

        s_prev = s
        a_prev = a

        if i % 100 == 0:
            scaler.epsilon_decay()

    return env    

def offline_sarsa(scaler, env_config, rl, mode):
    env = Environment(env_config)

    s = None
    a = None
    r_prev = None
    s_prev = None
    a_prev = None

    obs = env.next_state()
    s_prev, r_prev = scaler.monitoring(obs)
    a_prev = scaler.get_action(s_prev)

    for i in range(100000):
        if rl == "sarsa":
            env.scale_to(scaler.action_to_desired(a_prev))

        obs = env.next_state()
        if obs == False:
            # print("max iter : " + str(i))
            break

        s, r_prev = scaler.monitoring(obs)
        a = scaler.get_action(s)
        
        if mode == "train":
            scaler.update(s_prev, a_prev, r_prev, s, a)

        s_prev = s
        a_prev = a

    return env

def online_sarsa(scaler, env_config, rl):
    scaler.epsilon = 1.0
    env = Environment(env_config)

    s = None
    a = None
    r_prev = None
    s_prev = None
    a_prev = None

    obs = env.next_state()
    s_prev, r_prev = scaler.monitoring(obs)
    a_prev = scaler.get_action_with_noisy(s_prev)

    for i in range(100000):
        if rl == "sarsa":
            env.scale_to(scaler.action_to_desired(a_prev))

        obs = env.next_state()
        if obs == False:
            # print("max iter : " + str(i))
            break

        s, r_prev = scaler.monitoring(obs)
        a = scaler.get_action_with_noisy(s)
        
        scaler.update(s_prev, a_prev, r_prev, s, a)

        s_prev = s
        a_prev = a

        if i % 100 == 0:
            scaler.epsilon_decay()

    return env

def run_HPA_Q_Learning(test_env_config, train_env_config, rl_config, rl):
    scaler = None
    if rl == "q_learning":
        scaler = HPA_Q_Learning(rl_config)
    elif rl == "q_leanring_fixed_action":
        scaler = HPA_Q_Learning_Fixed_Action(rl_config)
    elif rl == "q_learning_trend":
        scaler = HPA_Q_Learning_Trend(rl_config)
    else:
        print(f"no such rl algorithm : {rl}")
        return

    env = None
    x = []
    result_history = []

    for j in range(1, 101):
        print("train : "+str(j))
        train_env = offline_q_learning(scaler, train_env_config, rl, "train")
        eval_env = offline_q_learning(scaler, test_env_config, rl, "eval")

        scaler.epsilon_decay()
        _, _, cost, _ = eval_env.get_result()
        result_history.append(cost)
        x.append(j)

        eval_env.print_result()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, result_history, 'r', label='total rewards')
    plt.show()

    print("last : "+str(j))
    env = online_q_learning(scaler, test_env_config, rl)
    env.plot_result()

    e, r, c, u, it, A, E, C = env.get_everything()
    result = [e, r, c, u]

    now = datetime.datetime.now()
    nowTime = now.strftime('%H_%M_%S')
    filename = rl+'_'+str(test["rate"])+'_t'+nowTime

    with open(f'./result/test4/{filename}.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(result_history)
        wr.writerow(result)
        wr.writerow(it)
        wr.writerow(A)
        wr.writerow(E)
        wr.writerow(C)

def run_HPA_SARSA(test_env_config, train_env_config, rl_config, rl):
    scaler = None
    if rl == "sarsa":
        scaler = HPA_SARSA(rl_config)
    else:
        print(f"no such rl algorithm : {rl}")
        return

    env = None
    x = []
    result_history = []

    for j in range(1, 101):
        print("train : "+str(j))
        train_env = offline_sarsa(scaler, train_env_config, rl, "train")
        eval_env = offline_sarsa(scaler, test_env_config, rl, "eval")

        scaler.epsilon_decay()
        _, _, cost, _ = eval_env.get_result()
        result_history.append(cost)
        x.append(j)

        eval_env.print_result()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, result_history, 'r', label='total rewards')
    plt.show()

    print("last : "+str(j))
    env = online_sarsa(scaler, test_env_config, rl)
    env.plot_result()

    e, r, c, u, it, A, E, C = env.get_everything()
    result = [e, r, c, u]

    now = datetime.datetime.now()
    nowTime = now.strftime('%H_%M_%S')
    filename = rl+'_'+str(test["rate"])+'_t'+nowTime

    with open(f'./result/test4/{filename}.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(result_history)
        wr.writerow(result)
        wr.writerow(it)
        wr.writerow(A)
        wr.writerow(E)
        wr.writerow(C)

def run_HPA_Q_Learning_v3(test_env_config, train_env_config, rl_config):
    scaler = HPA_Q_Learning_v3(rl_config)

    env = None

    x = []
    result_history = []

    for j in range(1000):
        print("train : "+str(j))
        env = Environment(train_env_config)

        s = None
        a = None
        r = None
        s_prev = None
        a_prev = None

        for i in range(100000):
            obs = env.next_state()
            if obs == False:
                # print("max iter : " + str(i))
                break
            s, r = scaler.convert_obs(obs)
            a = scaler.get_action(s)
            env.scale_to(a)

            if i > 0:
                scaler.update(s_prev, a_prev, s, r)

            s_prev = s
            a_prev = a

        scaler.epsilon_decay()
    
        env = Environment(test_env_config)

        s = None
        a = None
        r = None
        s_prev = None
        a_prev = None

        for i in range(10000):
            obs = env.next_state()
            if obs == False:
                # print("max iter : " + str(i))
                break
            s, r = scaler.convert_obs(obs)
            a = scaler.get_action(s)
            env.scale_to(a)

            s_prev = s
            a_prev = a

        result_history.append(env.print_result())
        x.append(j)

        if j % 100 == 0:
            env.plot_result()
        else:
            env.print_result()

        if j % 100 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(x, result_history, 'r', label='total rewards')
            plt.show()
        
        scaler.epsilon_decay()

    print("last : "+str(j))
    env = Environment(test_env_config)

    s = None
    a = None
    r = None
    s_prev = None
    a_prev = None

    for i in range(10000):
        if i < 200:
            env.epsilon = 0.3
        else:
            env.epsilon = 0
        obs = env.next_state()
        if obs == False:
            # print("test max iter : " + str(i))
            break
        s, r = scaler.convert_obs(obs)
        a = scaler.get_action(s)
        env.scale_to(a)

        if i > 0:
            diff = scaler.update(s_prev, a_prev, s, r)

        s_prev = s
        a_prev = a
        
    # simulator.print_result()
    # env.save_result('./result/advanced.csv')
    env.plot_result()

def get_history(path):
    history = []
    f = open(path, 'r', encoding='utf-8')
    rdr =csv.reader(f)
    for line in rdr:
        history.append(int(line[0]))

    print(f"{path} : "+str(len(history)))
    return history

if __name__ == '__main__':
    test = {}
    test["autoscale_period"] = 15
    test["rate"] = 20
    test["resource_cost"] = 0.000001389
    test["violation_cost"] = test["rate"] * test["resource_cost"]
    test["sim_period"] = 0.1
    test["timeout"] = 0.3
    test["scaling_delay"] = 5
    test["learning_rate"] = 0.05
    test["discount_factor"] = 0.5
    test["epsilon"] = 1.0
    test["min"] = 1
    test["max"] = 10
    
    test_history = get_history('./data/nasa-http-data_v3_test.csv')
    train_history = get_history('./data/nasa-http-data_v3_train.csv')
    history = get_history('./data/nasa-http-data_v3.csv')

    test_env_config = EnvConfig(
        application_profile=application_profile, 
        traffic_history=history[2001:], 
        init_pods=test["min"],
        timeout=test["timeout"], 
        autoscale_period=test["autoscale_period"], 
        simulation_period=test["sim_period"], 
        readiness_probe=test["scaling_delay"],
        resource_cost=test["resource_cost"],
        violation_cost=test["violation_cost"]
    )

    train_env_config = EnvConfig(
        application_profile=application_profile, 
        traffic_history=history[:2001], 
        init_pods=test["min"],
        timeout=test["timeout"], 
        autoscale_period=test["autoscale_period"], 
        simulation_period=test["sim_period"], 
        readiness_probe=test["scaling_delay"],
        resource_cost=test["resource_cost"],
        violation_cost=test["violation_cost"]
    )

    rl_config = RlConfig(
        pods_min=test["min"], 
        pods_max=test["max"], 
        resource_cost=test["resource_cost"], 
        violation_cost=test["violation_cost"], 
        autoscaling_period=test["autoscale_period"], 
        learning_rate=test["learning_rate"], 
        discount_factor=test["discount_factor"],
        epsilon=test["epsilon"]
    )

    target = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    for i in target:
        print(f'i: {i}')
        hpa_config = HpaConfig(
            pods_min=1, 
            pods_max=10, 
            target_utilization=i,
            scaling_tolerance=0.1
        )
   
        run_HPA(test_env_config, hpa_config)

    # run_HPA_SARSA(test_env_config, train_env_config, rl_config, 'sarsa')
    # run_HPA_Q_Learning(test_env_config, train_env_config, rl_config, 'q_learning')
    # run_HPA_Q_Learning(test_env_config, train_env_config, rl_config, 'q_leanring_fixed_action')
    # run_HPA_Q_Learning(test_env_config, train_env_config, rl_config, 'q_learning_trend')

    print("rate :" + str(test["rate"]))



import sys
import gym
import numpy as np
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import matplotlib.pyplot as plt
from pprint import pprint
from gym.envs.toy_text.frozen_lake import generate_random_map
import re
import random

random.seed(0)
np.random.seed(0)


def convert_PR(env):
    """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
    """
    transitions = env.P
    actions = int(re.findall(r'\d+', str(env.action_space))[0])
    states = int(re.findall(r'\d+', str(env.observation_space))[0])
    P = np.zeros((actions, states, states))
    R = np.zeros((states, actions))

    for state in range(states):
        for action in range(actions):
            for i in range(len(transitions[state][action])):
                tran_prob = transitions[state][action][i][0]
                state_ = transitions[state][action][i][1]
                R[state][action] += tran_prob*transitions[state][action][i][2]
                P[action, state, state_] += tran_prob

    return P, R


def get_stats_list(stats):
    errors = [stat['Error'] for stat in stats]
    reward = [stat['Reward'] for stat in stats]
    mean_v = [stat['Mean V'] for stat in stats]
    iterations = [stat['Iteration'] for stat in stats]
    times = [stat['Time'] for stat in stats]
    return iterations, errors, reward, mean_v, times


def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(
        np.mean(steps_list)))
    print('And you fell in the hole {:.2f} % of the times'.format(
        (misses/episodes) * 100))
    print('----------------------------------------------')


def check_if_new_episode(old_s, action, new_s):
    # print('check_if_new_episode', old_s, action, new_s)
    states = env.desc.astype(str).flatten()
    return states[new_s] == 'G' or states[new_s] == 'H'


def run_frozen_lake_ql(size=(4, 4), value=0.99, gamma=0.9, epsilon=0.999, epsilon_decay=0.99,
                       alpha_decay=1.0, alpha=0.6, p='gamma'):
    size_str = f"{size[0]}x{size[1]}"
    print("\n Running frozen lake mdp example QLearning")

    # Obtain P/R
    global env
    if size[0] == 4 and size[1] == 4:
        from gym.wrappers.time_limit import TimeLimit
        env = gym.make("forest-v1").env  # to avoid time limit
        env = TimeLimit(env, max_episode_steps=1000)
    else:
        random_map = generate_random_map(size=size[0], p=0.90)
        print(random_map)
        env = gym.make("forest-v1", desc=random_map).env

    env.seed(0)
    P, R = convert_PR(env)

    print(" starting QLearning")

    qlearn = hiive.mdptoolbox.mdp.QLearning(
        P, R, gamma=0.9, epsilon=epsilon, epsilon_min=0.1, n_iter=500000, iter_callback=check_if_new_episode)

    qlearn.setVerbose()
    stats = qlearn.run()
    print(stats)
    get_score(env, qlearn.policy)

    iterations, errors, reward, mean_v, times = get_stats_list(stats)
    plt.figure(10)
    plt.plot(iterations, times, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.title(f"forest {size_str} QL Time vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forest_ql_{size_str}_time_vs_iteration.png")

    plt.figure(11)
    plt.plot(iterations, errors, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(f"forest {size_str} QL Error vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forest_ql_{size_str}_error_vs_iteration.png")

    plt.figure(13)
    plt.plot(iterations, mean_v, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.title(f"forest {size_str} QL Mean V vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{p}_forest_ql_{size_str}_meanv_vs_iteration.png")

    return stats


def run_forest(size=10, max_iter=10000, algorithm='VI', gamma=0.99):
    size_str = f"{size}"

    print(f"\n Running forest example-{size_str}")

    P, R = hiive.mdptoolbox.example.forest(S=size)

    # Default episilon threshold
    # Define convergence
    if algorithm == 'PI':
        print("Starting policy iteration")
        policy_iter = hiive.mdptoolbox.mdp.PolicyIteration(
            P, R, gamma, max_iter=max_iter, eval_type=1)
        stats = policy_iter.run()
        print(policy_iter.policy)
        #get_score(env, policy_iter.policy)
    elif algorithm == 'VI':
        print("Starting value iteration")
        val_iter = hiive.mdptoolbox.mdp.ValueIteration(
            P, R, gamma, max_iter=max_iter)
        stats = val_iter.run()
        print(val_iter.policy)
        #get_score(env, val_iter.policy)

    # print stats
    iterations, errors, reward, mean_v, times = get_stats_list(stats)
    plt.figure(10)
    plt.plot(iterations, times, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.title(f"forest {size_str} {algorithm} Time vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{algorithm}_forest_{size_str}_time_vs_iteration.png")

    plt.figure(11)
    plt.plot(iterations, errors, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(f"forest {size_str} {algorithm} Error vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{algorithm}_forest_{size_str}_error_vs_iteration.png")

    plt.figure(12)
    plt.plot(iterations, reward, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title(f"forest {size_str} {algorithm} Reward vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{algorithm}_forest_{size_str}_reward_vs_iteration.png")

    plt.figure(13)
    plt.plot(iterations, mean_v, "-o", label=f'{gamma}')
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.title(f"forest {size_str} {algorithm} Mean V vs Iteration")
    plt.legend(loc="upper right")
    plt.savefig(f"{algorithm}_forest_{size_str}_meanv_vs_iteration.png")


# default args baby

# run_forest(10)

gammas = [0.1, 0.25, 0.5, 0.75, 0.99]
for gamma in gammas:
    # run_forest(size=10, algorithm="VI", gamma=gamma)
    # run_forest(size=10, algorithm="PI", gamma=gamma)
    # run_forest(size=1000, algorithm="VI", gamma=gamma)
    run_forest(size=1000, algorithm="PI", gamma=gamma)

# run_frozen_lake_ql()

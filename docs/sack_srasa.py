# #######################################################################
# # Copyright (C)                                                       #
# # 2020 Hongjia Liu(piperliu@qq.com)                                   #
# # Permission given to modify the code as long as you keep this        #
# # declaration at the top                                              #
# #######################################################################
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

class SackEnv:
    def __init__(self):
        self.day = 20
        self.homework = 10
        self.action_space = [0, 1]
    
    def step(self, action):
        # 0 homework
        # 1 play
        self.day -= 1
        reward = 0
        done = False

        assert action in [0, 1], 'Action Unknown!'

        if action == 0:
            reward = -5
            self.homework = max(0, self.homework - 1)
        if action == 1:
            reward = 10

        if self.day <= 0:
            done = True
            if self.homework > 0:
                reward = -1000
            else:
                reward = 0
        return (self.day, self.homework), reward, done, {}
    
    def reset(self):
        self.__init__()
        return (self.day, self.homework)

class SarsaAgent:
    def __init__(self, env):
        self.alpha = 0.1
        self.epsilon = 0.9
        self.gamma = 0.9

        self.q_value = {}
        self.env = env

    def train(self, run=1000, episode=200):
        rewards = np.zeros((run, episode))
        for r in trange(run):
            for e in range(episode):
                epsilon = self.epsilon / (e + 1)
                # epsilon = self.epsilon

                obs = self.env.reset()

                self.q_value.setdefault((obs, 0), 0)
                self.q_value.setdefault((obs, 1), 0)
                if np.random.binomial(1, epsilon) == 1:
                    action = np.random.choice(self.env.action_space)
                else:
                    if self.q_value[(obs, 0)] > self.q_value[(obs, 1)]:
                        action = 0
                    else:
                        action = 1
                while True:
                    q_value = self.q_value[(obs, action)]

                    obs_, reward, done, _ = self.env.step(action)

                    self.q_value.setdefault((obs_, 0), 0)
                    self.q_value.setdefault((obs_, 1), 0)
                    if np.random.binomial(1, epsilon) == 1:
                        action_ = np.random.choice(self.env.action_space)
                    else:
                        if self.q_value[(obs_, 0)] > self.q_value[(obs_, 1)]:
                            action_ = 0
                        else:
                            action_ = 1
                    if done:
                        self.q_value[(obs_, action_)] = 0
                    q_value_ = self.q_value[(obs_, action_)]
                    
                    rewards[r, e] += reward
                    self.q_value[(obs, action)] = \
                            q_value + self.alpha * \
                            (reward + self.gamma * q_value_ - q_value)
                    
                    if done:
                        break
                    
                    obs = obs_
                    action = action_
        return rewards

if __name__ == "__main__":
    sack_env = SackEnv()
    sarsa_agent = SarsaAgent(env=sack_env)
    rewards = sarsa_agent.train()
    plt.plot(rewards.mean(axis=0))
    plt.show()
    q_value = sarsa_agent.q_value
    print(q_value)

    q_work_table = np.zeros((20, 11))
    q_play_table = np.zeros((20, 11))
    q_optim_table = np.zeros((20, 11))
    for day in range(1, 21, 1):
        for homework in range(0, 11, 1):
            for action in [0, 1]:
                state = (day, homework)
                q_value.setdefault((state, action), 0)
                if action == 0:
                    q_work_table[day-1, homework] = q_value[(state, action)]
                if action == 1:
                    q_play_table[day-1, homework] = q_value[(state, action)]
            if q_work_table[day-1, homework] == q_play_table[day-1, homework]:
                q_optim_table[day-1, homework] = 0
            if q_work_table[day-1, homework] > q_play_table[day-1, homework]:
                q_optim_table[day-1, homework] = -10
            else:
                q_optim_table[day-1, homework] = 10

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    cmap = sns.cubehelix_palette(as_cmap=True)

    sns.heatmap(q_work_table, ax=ax1, cmap=cmap)
    ax1.set_title('work value')
    ax1.set_xlabel('homework left')
    ax1.set_ylabel('day')
    ax1.set_yticklabels(list(range(1, 21, 1)))

    sns.heatmap(q_play_table, ax=ax2, cmap=cmap)
    ax2.set_title('play value')
    ax2.set_xlabel('homework left')
    ax2.set_ylabel('day')
    ax2.set_yticklabels(list(range(1, 21, 1)))

    sns.heatmap(q_optim_table, ax=ax3, cmap=cmap)
    ax3.set_title('optimal choice')
    ax3.set_xlabel('homework left')
    ax3.set_ylabel('day')
    ax3.set_yticklabels(list(range(1, 21, 1)))
    plt.show()

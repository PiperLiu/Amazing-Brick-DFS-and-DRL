import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

import tianshou as ts
from DQN_train.gym_wrapper import AmazingBrickEnv2
from DQN_train.dqn_train2 import policy, torch, dqn2_path

env = AmazingBrickEnv2()
policy.load_state_dict(torch.load(dqn2_path + 'dqn2.pth'))

collector = ts.data.Collector(policy, env)
collector.collect(n_episode=10, render=1 / 20)
collector.close()
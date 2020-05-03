import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

import tianshou as ts
from DQN_train.gym_wrapper import AmazingBrickEnv3
from DQN_train.dqn_train3 import policy, torch, dqn3_path
import argparse

parser = argparse.ArgumentParser(description='choose which weights & bias to load')
parser.add_argument('pth_file')
parser.add_argument('--slay', action='store_true', default=False)
pth_name = 'dqn3'
args = parser.parse_args()
pth_name = pth_name + 'round_' + args.pth_file

try:
    policy.load_state_dict(torch.load(dqn3_path + pth_name + '.pth'))
except FileNotFoundError:
    print('this pth number does not exist!')

env = AmazingBrickEnv3(slay_okey=args.slay)

collector = ts.data.Collector(policy, env)
collector.collect(n_episode=10, render=1 / 60)
collector.close()

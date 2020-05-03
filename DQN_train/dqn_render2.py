import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

import tianshou as ts
from DQN_train.gym_wrapper import AmazingBrickEnv2
from DQN_train.dqn_train2 import policy, torch, dqn2_path
import argparse

parser = argparse.ArgumentParser(description='choose which weights & bias to load')
parser.add_argument('pth_file', choices=['0', '7', '10', '13', '21', '37', '40', '47'], help='0 is the latest pth')
pth_name = 'dqn2'
args = parser.parse_args()
if args.pth_file == '0':
    pass
else:
    pth_name = pth_name + 'round_' + args.pth_file

env = AmazingBrickEnv2()
policy.load_state_dict(torch.load(dqn2_path + pth_name + '.pth'))

collector = ts.data.Collector(policy, env)
collector.collect(n_episode=10, render=1 / 20)
collector.close()
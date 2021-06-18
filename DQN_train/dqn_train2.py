import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import GameState
from amazing_brick.game.amazing_brick_utils import CONST
from DQN_train.gym_wrapper import AmazingBrickEnv2

import tianshou as ts
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import json
import datetime

train_env = AmazingBrickEnv2()
test_env = AmazingBrickEnv2()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, state

state_shape = 28
action_shape = 1

net = Net()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)


'''args for rl'''
estimation_step = 3
max_epoch = 10
step_per_epoch = 300
collect_per_step = 50


policy = ts.policy.DQNPolicy(net, optim,
    discount_factor=0.9, estimation_step=estimation_step,
    target_update_freq=320)

train_collector = ts.data.Collector(policy, train_env, ts.data.ReplayBuffer(size=2000))
test_collector = ts.data.Collector(policy, test_env)

dqn2_path = osp.join(path, 'DQN_train/dqn_weights/')

if __name__ == '__main__':

    round = 0
    try:
        policy.load_state_dict(torch.load(dqn2_path + 'dqn2.pth'))
        lines = []
        with open(dqn2_path + 'dqn2_log.json', "r") as f:
            for line in f.readlines():
                cur_dict = json.loads(line)
                lines.append(cur_dict)
        log_dict = lines[-1]
        print(log_dict)
        round = log_dict['round']
        del lines
    except FileNotFoundError as identifier:
        print('\n\nWe shall train a bright new net.\n')
        pass
    while True:
        round += 1
        print('\n\nround:{}\n\n'.format(round))
        

        result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector,
            max_epoch=max_epoch, step_per_epoch=step_per_epoch,
            collect_per_step=collect_per_step,
            episode_per_test=30, batch_size=64,
            train_fn=lambda e1, e2: policy.set_eps(0.1 * (max_epoch - e1) / round),
            test_fn=lambda e1, e2: policy.set_eps(0.05 * (max_epoch - e1) / round), writer=None)
        print(f'Finished training! Use {result["duration"]}')

        torch.save(policy.state_dict(), dqn2_path + 'dqn2.pth')
        policy.load_state_dict(torch.load(dqn2_path + 'dqn2.pth'))
        
        log_dict = {}
        log_dict['round'] = round
        log_dict['last_train_time'] = datetime.datetime.now().strftime('%y-%m-%d %I:%M:%S %p %a')
        log_dict['result'] = json.dumps(result)
        with open(dqn2_path + 'dqn2_log.json', "a+") as f:
            f.write('\n')
            json.dump(log_dict, f)
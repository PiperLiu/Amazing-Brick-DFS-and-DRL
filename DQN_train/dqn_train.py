import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import GameState
from amazing_brick.game.amazing_brick_utils import CONST
from DQN_train.gym_wrapper import AmazingBrickEnv

import tianshou as ts
import torch, numpy as np
from torch import nn
import torch.nn.functional as F

train_env = AmazingBrickEnv(fps=1000)
test_env = AmazingBrickEnv(fps=1000)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 8, 4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 3)
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            # turn NHWC to NCHW
            obs = obs.permute(0, 3, 1, 2)
        x = F.max_pool2d(F.relu(self.conv1(obs)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, state

# state_shape = env.observation_space.shape or env.observation_space.n
# action_shape = env.action_space.shape or env.action_space.n
# net = Net(state_shape, action_shape)
# optim = torch.optim.Adam(net.parameters(), lr=1e-3)

state_shape = (80, 80, 4)
action_shape = 1

net = Net()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim,
    discount_factor=0.9, estimation_step=3,
    target_update_freq=320)

train_collector = ts.data.Collector(policy, train_env, ts.data.ReplayBuffer(size=200))
test_collector = ts.data.Collector(policy, test_env)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=1000, collect_per_step=10,
    episode_per_test=100, batch_size=64,
    train_fn=lambda e1, e2: policy.set_eps(0.1),
    test_fn=lambda e1, e2: policy.set_eps(0.05), writer=None)
print(f'Finished training! Use {result["duration"]}')

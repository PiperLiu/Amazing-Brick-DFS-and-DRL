# 构造一个简单的卷积神经网络，实现 DQN
本文涉及的 `.py` 文件有：
```
DQN_train/gym_warpper.py
DQN_train/dqn_train.py
```

## requirements
```
tianshou
pytorch > 1.40
gym
openCV
```

## 封装交互环境
强化学习算法有效，很大程度上取决于奖励机制设计的是否合理。

|事件|奖励|
|---|---|
|动作后碰撞障碍物、墙壁|-1|
|动作后无事发生|0.1|
|动作后得分|1|

封装代码在 [gym_wrapper.py](../DQN_train/gym_wrapper.py) 中，使用类 `AmazingBrickEnv` 。

## 强化学习机制与神经网络的构建
我设计的机制为：
- 每 2 帧进行一次动作决策；
- 状态的描述变量为 2 帧的图像。

对于每帧的图像处理如下。
```python
# 首先把图像转换成 RGB 矩阵
pygame.surfarray.array3d(pygame.display.get_surface())
# 使用 openCV 将 RGB 矩阵矩阵转换成 100*100 的灰度0-1矩阵
x_t = cv2.cvtColor(cv2.resize(obs, (100, 100)), cv2.COLOR_BGR2GRAY)
```

最后使用 `np.stack()` 将两帧数据合并，我们就得到了一个 2 通道的图像矩阵数据。

### 卷积神经网络的构建
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(通道数, 输出通道数, 卷积核大小, 步长)
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
```

神经网络解构如上述代码。

![../images/small-clip.gif](../images/small-clip.gif)

卷积训练过程如上图右。

### DQN 构建
```python
import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import GameState
from amazing_brick.game.amazing_brick_utils import CONST
from DQN_train.gym_wrapper import AmazingBrickEnv

# 使用了清华开源深度强化学习框架
import tianshou as ts
import torch, numpy as np
from torch import nn
import torch.nn.functional as F

train_env = AmazingBrickEnv(fps=1000)
test_env = AmazingBrickEnv(fps=1000)

state_shape = (80, 80, 4)
action_shape = 1

net = Net()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim,
    discount_factor=0.9, estimation_step=3,
    use_target_network=True, target_update_freq=320)

train_collector = ts.data.Collector(policy, train_env, ts.data.ReplayBuffer(size=200))
test_collector = ts.data.Collector(policy, test_env)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=1000, collect_per_step=10,
    episode_per_test=100, batch_size=64,
    train_fn=lambda e: policy.set_eps(0.1),
    test_fn=lambda e: policy.set_eps(0.05), writer=None)
print(f'Finished training! Use {result["duration"]}')
```

由于我还没有开始系统学习 NNs ，不了解 CNNs ，因此不是很信任自己建立的这个网络，没有投入资源与时间训练。

下两节（也是本项目的最后两节），我们将探讨线性网络解决这个控制问题的，其中将涉及到简单的建模与奖励机制设计讨论，会很有趣。

项目地址：[https://github.com/PiperLiu/Amazing-Brick-DFS-and-DRL](https://github.com/PiperLiu/Amazing-Brick-DFS-and-DRL)

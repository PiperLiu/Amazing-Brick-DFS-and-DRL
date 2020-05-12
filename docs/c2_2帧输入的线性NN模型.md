# 构造一个简单的神经网络，实现 DQN
本文涉及的 `.py` 文件有：
```
DQN_train/gym_warpper.py
DQN_train/dqn_train2.py
DQN_train/dqn_render2.py
```

## requirements
```
tianshou
pytorch > 1.40
gym
```

## 继续训练与测试
在本项目地址中，你可以使用如下文件对我训练的模型进行测试，或者继续训练。

#### 继续训练该模型
```bash
python DQN_train/dqn_train2.py
```
![dqn2_train](../images/dqn2_trian.png)
如图，我已经训练了 53 次（每次10个epoch），输入上述命令，你将开始第 54 次训练，如果不使用任务管理器强制停止，计算机将一直训练下去，并自动保存最新一代的权重。

#### 查看效果
```bash
python DQN_train/dqn_render2.py 0
```

注意参数 0 ，**输入 0 代表使用最新的权重。**

效果如图：
![dqn2_render](../images/dqn2_render.gif)
上图中，可以看到我们的 AI 已经学会了一些“知识”：比如如何前往下一层；它还需要多加练习，以学会如何避开这些小方块构成的障碍。

此外，我保留了一些历史权重。你还可以输入参数：7, 10, 13, 21, 37, 40, 47，查看训练次数较少时，神经网络的表现。

## 封装交互环境
强化学习算法有效，很大程度上取决于奖励机制设计的是否合理。

|事件|奖励|
|---|---|
|动作后碰撞障碍物、墙壁|-1|
|动作后无事发生|0.1|
|动作后得分|1|

封装代码在 [gym_wrapper.py](../DQN_train/gym_wrapper.py) 中，使用类 `AmazingBrickEnv2` 。

## 强化学习机制与神经网络的构建
上节中，我们将 2 帧的数据输入到卷积层中，目的是：
- 让卷积层提取出“障碍物边缘”与“玩家位置”；
- 让 2 帧数据反映出“玩家速度”信息。

为了节省计算资源，同时加快训练速度，我们人为地替机器提取这些信息：
- 不再将巨大的 2 帧“图像矩阵”输入到网络中；
- 取而代之的是，输入 2 帧的位置信息；
- 即输入`玩家xy坐标`、`左障碍物右上顶点xy坐标`、`右障碍物左上顶点xy坐标`、`4个障碍方块的左上顶点的xy坐标`（共14个数）；
- 如此， 2 帧数据共 28 个数字，我们的神经网络输入层只有 28 个神经元，比上一个模型（25600）少了不止一个数量级。

我设计的机制为：
- 每 2 帧进行一次动作决策；
- 状态的描述变量为 2 帧的图像。

#### 线性神经网络的构建
```python
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
```
如上，共四层线性网络。

## 记录训练的微型框架
为了保存训练好的权重，且在需要时可以暂停并继续训练，我新建了一个`.json`文件用于保存训练数据。
```python
dqn2_path = osp.join(path, 'DQN_train/dqn_weights/')

if __name__ == '__main__':

    round = 0
    try:
        # 此处 policy 采用 DQN
        # 具体 DQN 构建方法见下文
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
            train_fn=lambda e: policy.set_eps(0.1 * (max_epoch - e) / round),
            test_fn=lambda e: policy.set_eps(0.05 * (max_epoch - e) / round), writer=None)
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
```

## DQN
```python
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
    use_target_network=True, target_update_freq=320)

train_collector = ts.data.Collector(policy, train_env, ts.data.ReplayBuffer(size=2000))
test_collector = ts.data.Collector(policy, test_env)

```

![dqn2_render](../images/dqn2_render.gif)
如图，采用这种方式训练了 53 个循环（共计 53 * 10 * 300 = 159000 个 epoch）效果还是一般。

下一节（也是本项目的最后一节），我们将探讨线性网络解决这个控制问题的相对成功的方案。

项目地址：[https://github.com/PiperLiu/Amazing-Brick-DFS-and-DRL](https://github.com/PiperLiu/Amazing-Brick-DFS-and-DRL)

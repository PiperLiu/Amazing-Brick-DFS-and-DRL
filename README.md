# Amazing Brick with BFS & DRL
### Preface
去年在 B 站看到大佬 UP [The CW](https://space.bilibili.com/13081489) 的视频：[用AI在手游中作弊！内藏干货：神经网络、深度/强化学习讲解](https://www.bilibili.com/video/BV1Ft411E79Y)，当时觉得很有趣，但“强化学习”这部分看的半知半解。

未曾想，导师给我定的毕设研究方向也是“强化学习”。疫情期间，在家做完毕设实验后，由于拖延症晚期，实在不想动笔写论文（尤其是“选题背景”、“选题意义”这种...）。于是便想着先复现一下这位 UP 的**游戏+算法**，供小伙伴们更好地学习。

![./images/small-clip.gif](./images/small-clip.gif)

本项目包括：
- 基于 pygame 的 amazing-brick 游戏复现，可以在电脑端玩此小游戏；
- 基于 广度优先搜索算法(BFS, Breadth-First-Search) 的自动游戏机制；
- - 基于 宽度优先搜索算法(DFS, Depth-First-Search) 的自动游戏机制；
- 基于 清华开源强化学习库 tianshou 的 DQN 算法，以强化学习的方式在游戏中实现自动控制。

本项目参考或使用了如下资源：
- [The CW](https://space.bilibili.com/13081489) 的 Bilibili 视频：[用AI在手游中作弊！内藏干货：神经网络、深度/强化学习讲解](https://www.bilibili.com/video/BV1Ft411E79Y)
- [yenchenlin](https://github.com/yenchenlin) 的 [Flappy Bird 项目](https://github.com/yenchenlin/DeepLearningFlappyBird)

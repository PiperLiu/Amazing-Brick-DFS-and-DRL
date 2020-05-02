import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import GameState

import gym
import cv2
import numpy as np

class AmazingBrickEnv(gym.Env):
    def __init__(self, fps=200):
        self._step = 0
        self.max_step = 20000
        self.g_s = GameState(fps=fps)
        self.action_space = gym.spaces.Discrete(3)
    
    def step(self, action):
        done_ = False
        observation, reward, done = self.g_s.frame_step(0)
        x_t1, done_ = obs2x_t(observation, done, done_)
        cv2.imshow('window', x_t1)
        observation, reward, done = self.g_s.frame_step(action)
        x_t2, done_ = obs2x_t(observation, done, done_)
        cv2.imshow('window', x_t2)
        s_t = x_t2s_t(x_t1, x_t2)
        if self._step >= self.max_step:
            done = True
        self._step += 1
        return s_t, reward, done_, {}
    
    def reset(self):
        self.g_s.__init__()
        self._step = 0
        done_ = False
        observation, reward, done = self.g_s.frame_step(0)
        x_t1, done_ = obs2x_t(observation, done, done_)
        observation, reward, done = self.g_s.frame_step(0)
        x_t2, done_ = obs2x_t(observation, done, done_)
        s_t = x_t2s_t(x_t1, x_t2)
        return s_t
    
    def render(self, mode='human'):
        return
    

def obs2x_t(obs, done, done_):
    x_t = cv2.cvtColor(cv2.resize(obs, (100, 100)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    if done:
        done_ = done
    return x_t, done_

def x_t2s_t(*args):
    s_t = np.stack((args), axis=2)
    return s_t
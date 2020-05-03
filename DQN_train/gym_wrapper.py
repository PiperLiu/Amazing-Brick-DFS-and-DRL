import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import \
        GameState, SCREEN, BACKGROUND_COLOR, pygame, FPSCLOCK

import gym
import cv2
import numpy as np

class AmazingBrickEnv(gym.Env):
    def __init__(self, fps=200):
        self._step = 0
        self.max_score = 200
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
        if self.g_s.score >= self.max_score:
            self.reset()
            done = True
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

class AmazingBrickEnv2(gym.Env):
    def __init__(self):
        self.max_score = 200
        self.g_s = GameState(ifRender=False)
        self.action_space = gym.spaces.Discrete(3)
    
    def step(self, action):
        done_ = False
        observation, reward, done = self.g_s.frame_step(0)
        if done:
            done_ = done
        obs, reward, done = self.g_s.frame_step(action)
        if done:
            done_ = done
        observation.extend(obs)
        if self.g_s.score >= self.max_score:
            self.reset()
            done = True
        return np.asarray(observation), reward, done_, {}
    
    def reset(self):
        self.g_s.__init__(ifRender=False)
        done_ = False
        observation, reward, done = self.g_s.frame_step(0)
        if done:
            done_ = done
        obs, reward, done = self.g_s.frame_step(0)
        if done:
            done_ = done
        observation.extend(obs)
        return np.asarray(observation)
    
    def render(self, mode='human'):
        SCREEN.fill(BACKGROUND_COLOR)
        for pipe in self.g_s.pipes:
            self.g_s.s_c(pipe)
            pipe.draw(SCREEN)
        for block in self.g_s.blocks:
            self.g_s.s_c(block)
            block.draw(SCREEN)
        self.g_s.s_c(self.g_s.player)
        self.g_s.player.draw(SCREEN)
        self.g_s._showScore()
        observation = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(self.g_s.fps)
        return

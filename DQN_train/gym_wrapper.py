import os.path as osp
import sys
dirname = osp.dirname(__file__)
path = osp.join(dirname, '..')
sys.path.append(path)

from amazing_brick.game.wrapped_amazing_brick import \
        GameState, SCREEN, BACKGROUND_COLOR, pygame, FPSCLOCK

import gym
import numpy as np

class AmazingBrickEnv(gym.Env):
    def __init__(self, fps=200):
        import cv2
        self._step = 0
        self.max_score = 200
        self.fps = fps
        self.g_s = GameState(fps=self.fps)
        self.action_space = gym.spaces.Discrete(3)
    
    def step(self, action):
        import cv2
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
        self.g_s.__init__(fps=self.fps)
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
    import cv2
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
        self.ifRender = False
    
    def step(self, action):
        if self.ifRender:
            if action == 0:
                print('do nothing')
            elif action == 1:
                print('left')
            elif action == 2:
                print('right')
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
        self.ifRender = True
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

class AmazingBrickEnv3(gym.Env):
    def __init__(self, slay_okey=False):
        self.max_score = 200
        self.max_step = 500
        self._step = 0
        self.g_s = GameState(ifRender=False)
        self.action_space = gym.spaces.Discrete(3)
        self.slay_okey = slay_okey
        self.ifRender = False
    
    def step(self, action):
        if self.ifRender:
            if action == 0:
                print('do nothing')
            elif action == 1:
                print('left')
            elif action == 2:
                print('right')
        obs = []
        observation, reward, done = self.g_s.frame_step(action)
        for ind, o in enumerate(observation):
            if ind % 2 == 0:
                obs.append(o / self.g_s.s_c.width)
            else:
                obs.append(o / self.g_s.s_c.height)
        # velX, velY
        obs.append(self.g_s.player.velX / self.g_s.player.velMaxX)
        obs.append(self.g_s.player.velY / self.g_s.player.velMaxY)
        if self.g_s.score >= self.max_score:
            self.reset()
            done = True
        self._step += 1
        """
        I add `self._step >= self.max_step and self.g_s.score <= 1`
        to let the meaningless slay done,
        and it **whims** at epoch 3rd, howerver,
        it slay at epoch 4, 5, 6...
        So, I add reward to punish this meaningless slay.
        """
        if self._step >= self.max_step and self.g_s.score <= 1 and not self.slay_okey:
            self.reset()
            done = True
            if reward == 0.1:
                reward = - 10
        """
        dqn3round_1slay.pth
        dqn3round_2slay.pth
        dqn3round_3slay.pth
            are all very slay, they are just striving for life
        To avoid this, I let the reward==0.1 lower
        """
        if reward == 0.1:
            reward = 0.0001
        return np.asarray(obs), reward, done, {}
    
    def reset(self):
        self.g_s.__init__(ifRender=False)
        self._step = 0
        done = True
        while done:
            obs = []
            observation, reward, done = self.g_s.frame_step(0)
            for ind, o in enumerate(observation):
                if ind % 2 == 0:
                    obs.append(o / self.g_s.s_c.width)
                else:
                    obs.append(o / self.g_s.s_c.height)
            # velX, velY
            obs.append(self.g_s.player.velX / self.g_s.player.velMaxX)
            obs.append(self.g_s.player.velY / self.g_s.player.velMaxY)
        return np.asarray(obs)
    
    def render(self, mode='human'):
        self.ifRender = True
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
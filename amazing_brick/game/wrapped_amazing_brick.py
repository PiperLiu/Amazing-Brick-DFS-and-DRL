import numpy as np
import sys
import random
import pygame
from .amazing_brick_utils import CONST, load, Box, Player, Block, pipes
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((CONST['SCREEN_WIDTH'], CONST['SCREEN_HEIGHT']))
pygame.display.set_caption('Amazing Brick')

IMAGES = load()
BACKGROUND_COLOR = (240, 255, 240)

class ScreenCamera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = CONST['SCREEN_WIDTH']
        self.height = CONST['SCREEN_HEIGHT']
        self.x_ = self.x + self.width
        self.y_ = self.y + self.height
    
    def __call__(self, obj: Box):
        # output the obj's (x, y) on screen
        x_c = obj.x - self.x
        y_c = obj.y - self.y
        obj.set_camera(x_c, y_c)
        return obj
    
    def move(self, obj: Player):
        self(obj)
        if obj.y_c < self.height / 2:
            self.y -= (self.height / 2 - obj.y_c)
        else:
            pass

class GameState:
    def __init__(self, ifRender=True, fps=FPS):
        self.score = 0
        self.player = Player()
        self.pipes = []
        self.blocks = []
        self.s_c = ScreenCamera()
        self.ifRender = ifRender
        self.color_ind = 0

        self._getRandomPipe(init=True)

        self.fps = fps

    def frame_step(self, action):
        pygame.event.pump()

        reward = 0.1
        done = False

        assert action in [0, 1, 2], 'Action is not available!'
        # [stay, lFlap, rFlap]

        if action == 0:
            self.player.noneDo()
        elif action == 1:
            self.player.lFlap()
        elif action == 2:
            self.player.rFlap()
        
        # check for score
        playerMidPos = self.s_c(self.player).y_c + self.player.height / 2
        for ind, pipe in enumerate(self.pipes):
            if ind % 2 == 1:
                continue
            self.s_c(pipe)
            if pipe.y_c <= playerMidPos <= pipe.y_c + pipe.height:
                if not pipe.scored:
                    self.score += 1
                    pipe.scored = True
                    reward = 1
        
        # player's movement
        self.player.x += self.player.velX
        self.player.x_ += self.player.velX
        self.player.y += self.player.velY
        self.player.y_ += self.player.velY
        self.s_c.move(self.player)

        # add new pipe and remove pipe
        low_pipe = self.pipes[0]
        if self.s_c(low_pipe).y_c >= self.s_c.height - low_pipe.width \
                and len(self.pipes) < 6:
            self._getRandomPipe()
        if self.s_c(low_pipe).y_c >= self.s_c.height \
                and len(self.pipes) > 4:
            self.pipes.pop(0)
            self.pipes.pop(0)
        
        # remove blocks
        for block in self.blocks:
            self.s_c(block)
            x_flag = - CONST['BLOCK_WIDTH'] <= block.x_c <= self.s_c.width
            y_flag = block.y_c >= self.s_c.height
        
        # check if crash here
        if self.s_c(self.player).y_c + self.player.height >= self.s_c.height:
            self.__init__(ifRender=self.ifRender)
            done = True
            reward = -1
        if not done:
            if self.player.x_c <= self.s_c.x or \
                    self.player.x_c >= self.s_c.width - self.player.width:
                self.__init__(ifRender=self.ifRender)
                done = True
                reward = -1
        if not done:
            for pipe in self.pipes:
                if self.player.check_crash(pipe):
                    self.__init__(ifRender=self.ifRender)
                    done = True
                    reward = -1
        if not done:
            for block in self.blocks:
                if self.player.check_crash(block):
                    self.__init__(ifRender=self.ifRender)
                    done = True
                    reward = -1

        # draw
        if self.ifRender:
            SCREEN.fill(BACKGROUND_COLOR)
            for pipe in self.pipes:
                self.s_c(pipe)
                pipe.draw(SCREEN)
            for block in self.blocks:
                self.s_c(block)
                block.draw(SCREEN)
            self.s_c(self.player)
            self.player.draw(SCREEN)
            self._showScore()
            observation = pygame.surfarray.array3d(pygame.display.get_surface())
            pygame.display.update()
            FPSCLOCK.tick(self.fps)
        else:
            playerXc, playerYc = self.s_c(self.player).x_c, self.player.y_c
            pipe_Xc_Yc = []
            block_Xc_Yc = []
            for ind, pipe in enumerate(self.pipes):
                if pipe.scored:
                    continue
                if len(pipe_Xc_Yc) == 4:
                    break
                if ind % 2 == 0:
                    pipe_Xc, pipe_Yc = self.s_c(pipe).x_c + pipe.width, pipe.y_c
                else:
                    pipe_Xc, pipe_Yc = self.s_c(pipe).x_c, pipe.y_c
                pipe_Xc_Yc.extend([pipe_Xc, pipe_Yc])
            for block in self.blocks[:4]:
                block_Xc, block_Yc = self.s_c(block).x_c, block.y_c
                block_Xc_Yc.extend([block_Xc, block_Yc])
            observation = [playerXc, playerYc]
            observation.extend(pipe_Xc_Yc)
            observation.extend(block_Xc_Yc)

        return observation, reward, done

    def _getRandomPipe(self, init=False):
        if self.score % 5 == 4:
            self.color_ind = (self.color_ind + 1) % 5

        gap_left_topXs = list(range(100, 190, 20))
        if init:
            index = random.randint(0, len(gap_left_topXs)-1)
            x = gap_left_topXs[index]
            y = CONST['SCREEN_HEIGHT'] / 2 - CONST['PIPE_WIDTH'] / 2
            first_pipes = pipes(x, y, self.color_ind)
            self.pipes.append(first_pipes[0])
            self.pipes.append(first_pipes[1])
            self._addBlocks()
        index = random.randint(0, len(gap_left_topXs)-1)
        x = self.s_c.x + gap_left_topXs[index]
        y = self.pipes[-1].y - CONST['SCREEN_HEIGHT'] / 2
        pipe = pipes(x, y, self.color_ind)
        self.pipes.append(pipe[0])
        self.pipes.append(pipe[1])
        self._addBlocks()
    
    def _addBlocks(self):
        x = (self.pipes[-2].x_ + self.pipes[-1].x) / 2
        y = (self.pipes[-2].y + self.pipes[-2].y_) / 2
        for i in range(2, 0, -1):
            y_block = y + i * CONST['BLOCK_SPACE']
            x_block = x + np.random.normal() * CONST['PIPE_GAPSIZE'] / 2.5
            block = Block(x_block, y_block, self.color_ind)
            self.blocks.append(block)

    def _showScore(self):
        scoreDigits = [int(x) for x in list(str(self.score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += IMAGES['numbers'][digit].get_width()

        Xoffset = (CONST['SCREEN_WIDTH'] - totalWidth) / 2

        for digit in scoreDigits:
            SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, CONST['SCREEN_HEIGHT'] * 0.1))
            Xoffset += IMAGES['numbers'][digit].get_width()

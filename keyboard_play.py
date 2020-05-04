import os.path as osp
import sys
dirname = osp.dirname(__file__)
sys.path.append(dirname)

import pygame
from amazing_brick.game.wrapped_amazing_brick import \
        GameState, SCREEN

game_state = GameState(True)
ACTIONS = (0, 1, 2)

while True:
    action = ACTIONS[0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFTBRACKET:
                action = ACTIONS[1]
            if event.key == pygame.K_RIGHTBRACKET:
                action = ACTIONS[2]

    game_state.frame_step(action)
pygame.quit()

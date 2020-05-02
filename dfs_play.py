import os.path as osp
import sys
dirname = osp.dirname(__file__)
sys.path.append(dirname)

import pygame
import numpy as np
import time
from amazing_brick.game.wrapped_amazing_brick import \
        GameState, SCREEN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--display', action='store_true', default=False)
args = parser.parse_args()

game_state = GameState(True)
final_s_a_list = []

ACTIONS = (0, 1, 2)

def dfs_forward(root_state, show=False):
    global final_s_a_list
    final_s_a_list = []

    def dfs(state, s_a_list):
        global final_s_a_list
        # a trick
        if len(s_a_list) % 2 == 1:
            ACTIONS_tmp = (2, 1)
        else:
            ACTIONS_tmp = (1, 2)
        
        for action in ACTIONS_tmp:
            if len(final_s_a_list) > 0:
                break
            # new_state = move_forward(state, ACTIONS[0])
            new_state = move_forward(state, action)
            new_s_a_list = s_a_list.copy()
            new_s_a_list.append((new_state, action))
            if check_crash(new_state):
                if show:
                    pygame.draw.rect(SCREEN, (255, 0, 0), \
                            (new_state['x'] - game_state.s_c.x, new_state['y'] - game_state.s_c.y, game_state.player.width, game_state.player.height))
                    pygame.display.update()
                del new_state
                del new_s_a_list
            else:
                if show:
                    pygame.draw.rect(SCREEN, (100, 100, 100), \
                            (new_state['x'] - game_state.s_c.x, new_state['y'] - game_state.s_c.y, game_state.player.width, game_state.player.height))
                    pygame.display.update()
                if check_for_score(new_state):
                    if show:
                        pygame.draw.rect(SCREEN, (0, 0, 255), \
                                (new_state['x'] - game_state.s_c.x, new_state['y'] - game_state.s_c.y, game_state.player.width, game_state.player.height))
                        pygame.display.update()
                    final_s_a_list = new_s_a_list
                    break
                dfs(new_state, new_s_a_list)

    dfs(root_state, [])

    return final_s_a_list

def check_for_score(state):
    # check for score
    playerMidPos = state['y'] - game_state.s_c.y + game_state.player.height / 2
    for ind, pipe in enumerate(game_state.pipes):
        if ind % 2 == 1:
            continue
        game_state.s_c(pipe)
        if playerMidPos <= pipe.y_c:
            if not pipe.scored:
                return True
    return False

def move_forward(state, action):
    new_state = state.copy()
    if action == 0:
        if state['velX'] > 0:
            new_state['velX'] -= game_state.player.dragForce
        elif state['velX'] < 0:
            new_state['velX'] += game_state.player.dragForce
        new_state['velY'] += game_state.player.gravity
        if new_state['velX'] > game_state.player.velMaxX:
            new_state['velX'] = 10
        if new_state['velX'] < - game_state.player.velMaxX:
            new_state['velX'] = - 10
        if new_state['velY'] > game_state.player.velMaxX:
            new_state['velY'] = 10
        if new_state['velY'] < - game_state.player.velMaxX:
            new_state['velY'] = - 10
    elif action == 1:
        new_state['velX'] -= game_state.player.AccX
        new_state['velY'] -= (game_state.player.AccY - game_state.player.gravity)
        if new_state['velX'] > game_state.player.velMaxX:
            new_state['velX'] = 10
        if new_state['velX'] < - game_state.player.velMaxX:
            new_state['velX'] = - 10
        if new_state['velY'] > game_state.player.velMaxX:
            new_state['velY'] = 10
        if new_state['velY'] < - game_state.player.velMaxX:
            new_state['velY'] = - 10
    elif action == 2:
        new_state['velX'] += game_state.player.AccX
        new_state['velY'] -= (game_state.player.AccY - game_state.player.gravity)
        if new_state['velX'] > game_state.player.velMaxX:
            new_state['velX'] = 10
        if new_state['velX'] < - game_state.player.velMaxX:
            new_state['velX'] = - 10
        if new_state['velY'] > game_state.player.velMaxX:
            new_state['velY'] = 10
        if new_state['velY'] < - game_state.player.velMaxX:
            new_state['velY'] = - 10
    
    new_state['x'] += new_state['velX']
    new_state['y'] += new_state['velY']
    return new_state

def check_crash(state):
    # check if crash here
    x_c = state['x'] - game_state.s_c.x
    y_c = state['y'] - game_state.s_c.y
    if y_c + game_state.player.height >= game_state.s_c.height:
        return True
    if x_c <= game_state.s_c.x or \
            x_c >= game_state.s_c.width - game_state.player.width:
        return True
    for pipe in game_state.pipes:
        x0, y0, x0_, y0_ = state['x'], state['y'], \
                state['x'] + game_state.player.width, \
                state['y'] + game_state.player.height
        x1, y1, x1_, y1_ = pipe.box()
        lx = abs((x0 + x0_) / 2 - (x1 + x1_) / 2)
        ly = abs((y0 + y0_) / 2 - (y1 + y1_) / 2)
        if lx <= (game_state.player.width + pipe.width) / 2 and \
                ly <= (game_state.player.height + pipe.height) / 2:
            return True
    for block in game_state.blocks:
        x0, y0, x0_, y0_ = state['x'], state['y'], \
                state['x'] + game_state.player.width, \
                state['y'] + game_state.player.height
        x1, y1, x1_, y1_ = block.box()
        lx = abs((x0 + x0_) / 2 - (x1 + x1_) / 2)
        ly = abs((y0 + y0_) / 2 - (y1 + y1_) / 2)
        if lx <= (game_state.player.width + block.width) / 2 and \
                ly <= (game_state.player.height + block.height) / 2:
            return True
    return False

while True:
    game_state.frame_step(ACTIONS[0])
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    root_state = {}
    root_state['x'] = game_state.player.x
    root_state['y'] = game_state.player.y
    root_state['velX'] = game_state.player.velX
    root_state['velY'] = game_state.player.velY

    s_a_list = dfs_forward(root_state, args.display)
    if args.display:
        time.sleep(0.2)
    for s_a in s_a_list:
        action = s_a[1]
        # game_state.frame_step(ACTIONS[0])
        game_state.frame_step(action)
    
pygame.quit()

import pygame
import sys
import os.path as osp

path = osp.dirname(__file__)
path = osp.join(path, '..')

# const
CONST = {}
CONST['PLAYER_WIDTH'] = 14
CONST['PLAYER_HEIGHT'] = 14
CONST['BLOCK_WIDTH'] = 16
CONST['BLOCK_HEIGHT'] = 16
CONST['PIPE_WIDTH'] = 16
CONST['PIPE_GAPSIZE'] = 70
CONST['PIPE_SPACE'] = 245
CONST['SCREEN_WIDTH'] = 288
CONST['SCREEN_HEIGHT'] = 490
CONST['BLOCK_SPACE'] = CONST['SCREEN_HEIGHT'] / 2 / 3

CONST['FPS'] = 30

COLORS = (
    [218,112,214],
    [123,104,238],
    [143,188,143],
    [255,215,0],
    [169,169,169]
)


def load():

    IMAGES = {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load(path + '/assets/sprites/0.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/1.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/2.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/3.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/4.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/5.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/6.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/7.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/8.png').convert_alpha(),
        pygame.image.load(path + '/assets/sprites/9.png').convert_alpha()
    )
    return IMAGES

class Box:
    def __init__(self, x, y, x_, y_):
        # left top
        self.x = x
        self.y = y
        # right bottom
        self.x_ = x_
        self.y_ = y_
        self.width = self.x_ - self.x
        self.height = self.y_ - self.y

        # camera
        self.x_c = 0
        self.y_c = 0
    
    def box(self):
        return self.x, self.y, self.x_, self.y_
    
    def draw(self):
        raise NotImplementedError

    def set_camera(self, x_c, y_c):
        self.x_c = x_c
        self.y_c = y_c

class Player(Box):
    def __init__(self):
        super().__init__(
            x = CONST['SCREEN_WIDTH'] / 2 - CONST['PLAYER_WIDTH'] / 2,
            y = CONST['SCREEN_HEIGHT'] - 5 * CONST['PLAYER_HEIGHT'],
            x_= CONST['SCREEN_WIDTH'] / 2 + CONST['PLAYER_WIDTH'] / 2,
            y_= CONST['SCREEN_HEIGHT'] - 4 * CONST['PLAYER_HEIGHT'],
        )
        self.gravity = 0.35
        self.dragForce = 0.01
        self.velX = 0
        self.velY = 0
        self.AccX = 4.5
        self.AccY = 2.5
        self.velMaxX = 10
        self.velMaxY = 10
    
    def lFlap(self):
        self.velX -= self.AccX
        self.velY -= (self.AccY - self.gravity)
        self._checkMax()
    
    def rFlap(self):
        self.velX += self.AccX
        self.velY -= (self.AccY - self.gravity)
        self._checkMax()
    
    def noneDo(self):
        if self.velX > 0:
            self.velX -= self.dragForce
        elif self.velX < 0:
            self.velX += self.dragForce
        self.velY += self.gravity
        self._checkMax()
    
    def _checkMax(self):
        if self.velX > self.velMaxX:
            self.velX = 10
        if self.velX < - self.velMaxX:
            self.velX = - 10
        if self.velY > self.velMaxY:
            self.velY = 10
        if self.velY < - self.velMaxY:
            self.velY = - 10
    
    def check_crash(self, obj: Box):
        x0, y0, x0_, y0_ = self.box()
        x1, y1, x1_, y1_ = obj.box()
        lx = abs((x0 + x0_) / 2 - (x1 + x1_) / 2)
        ly = abs((y0 + y0_) / 2 - (y1 + y1_) / 2)
        if lx <= (self.width + obj.width) / 2 and \
                ly <= (self.height + obj.height) / 2:
            return True
        return False
    
    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), (self.x_c, self.y_c, self.width, self.height))

class Block(Box):
    def __init__(self, x, y, color_ind=0):
        super().__init__(x, y, x+CONST['BLOCK_WIDTH'], y+CONST['BLOCK_HEIGHT'])
        self.color = COLORS[color_ind]
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x_c, self.y_c, self.width, self.height))

class Pipe(Box):
    def __init__(self, x, y, x_, color_ind=0):
        super().__init__(x, y, x_, y+CONST['PIPE_WIDTH'])
        self.scored = False
        self.color = COLORS[color_ind]
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x_c, self.y_c, self.width, self.height))

def pipes(x, y, color_ind=0):
    # x, y is the left-top point of right-pipe
    right_pipe_x = x
    right_pipe_y = y
    right_pipe_x_= CONST['SCREEN_WIDTH']
    rPipe = Pipe(right_pipe_x, right_pipe_y, right_pipe_x_, color_ind)

    left_pipe_x = 0
    left_pipe_y = y
    left_pipe_x_= x - CONST['PIPE_GAPSIZE']
    lPipe = Pipe(left_pipe_x, left_pipe_y, left_pipe_x_, color_ind)

    return lPipe, rPipe

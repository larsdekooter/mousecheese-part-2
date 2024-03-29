import pygame
import time
from data import cooldown


class Mouse:
    def __init__(self, x, y):
        self.img = pygame.transform.scale(pygame.image.load("mouse.png"), (100, 100))
        self.x = x
        self.y = y
        self.noMove = False

    def move(self, move):
        if move[0] == 1 and self.y == 0:
            self.noMove = True
        elif move[1] == 1 and self.y == 700:
            self.noMove = True
        elif move[2] == 1 and self.x == 0:
            self.noMove = True
        elif move[3] == 1 and self.x == 700:
            self.noMove = True
        else:
            self.noMove = False
        if move[0] == 1 and self.y != 0:  # up
            self.y -= 100
        elif move[1] == 1 and self.y != 700:  # down
            self.y += 100
        elif move[2] == 1 and self.x != 0:  # left
            self.x -= 100
        elif move[3] == 1 and self.x != 700:  # right
            self.x += 100

import pygame
from data import catPositions, cheeseReward, getDistanceReward, rewardNerf, getEfficiencyPenalty, invalidMovePunishment
from mouse import Mouse
import math
import time

class Game:
    def __init__(self):
        self.mouse = Mouse(0, 0)
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.catimg = pygame.transform.scale(pygame.image.load("cat.png"), (100, 100))
        self.cheeseimg = pygame.transform.scale(pygame.image.load("OIP.jpg"), (100, 100))
        self.font = pygame.font.Font("arial.ttf", 32)
        self.distances = []
        self.gameTime = time.time()

    def reset(self):
        self.mouse = Mouse(0,0)
    
    def step(self, move):
        self.preLoad()
        self.load()
        self.mouse.move(move)
        self.afterLoad()
        done = self.checkDeath()
        reward, done, won = self.getReward(done)

        return done, reward, won

    def getReward(self, done):
        reward = 0
        distance = self.getDistanceToCheese()
        won = False
        if done:
            reward = -cheeseReward
        elif self.checkCheese():
            reward = cheeseReward
            done = True
            won = True
        else:
            reward = getDistanceReward(distance) - getEfficiencyPenalty(distance)
            if self.mouse.noMove:
                reward = invalidMovePunishment
        return reward, done, won

    def getDistanceToCheese(self):
        dx = 700 - self.mouse.x
        dy = 700 - self.mouse.y
        return math.sqrt(dx*dx + dy*dy)

    def checkCheese(self):
        if (self.mouse.x, self.mouse.y) == (700, 700):
            return True
        return False
    
    def checkDeath(self):
        if any(pos == (self.mouse.x, self.mouse.y) for pos in catPositions):
            return True
        return False

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
    def preLoad(self):
        self.handleEvents()
        self.screen.fill('black')

    def afterLoad(self):
        self.screen.blit(self.cheese, (700, 700))
        self.screen.blit(self.mouse.img, (self.mouse.x, self.mouse.y))
        for cat in self.cats:
            self.screen.blit(cat[0], cat[1])
        pygame.display.flip()
        self.clock.tick(60)

    def load(self):
        gap = 100
        for i in range(8):
            pygame.draw.line(self.screen, "red", ((i+1)*gap,0), ((i+1)*gap, 800))
        for i in range(8):
            pygame.draw.line(self.screen, "red", (0, (i+1)*gap), (1280, (i+1)*gap))

        self.cheese = self.cheeseimg
        self.cats = self.drawCats()

    def drawCats(self):
        cats = []
        for position in catPositions:
            cats.append([self.catimg, position])
        return cats
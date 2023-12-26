import torch
import random
import numpy as np
from collections import deque
from game import Game
from model import Linear_QNet, QTrainer
from helper import plot
import data
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(9, data.hiddenSize, 4)
        if os.path.exists("C:/Users/Kooter/Documents/VSC Projects/A.I/snake - kopie/model/model.pth"):
            self.model.load_state_dict(torch.load('C:/Users/Kooter/Documents/VSC Projects/A.I/snake - kopie/model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.decayStep = 0
        self.aiMoves = 0
        self.randomMoves = 0


    def get_state(self, game: Game):
        distanceToCheese = game.getDistanceToCheese()
        aroundLocations = [
            (game.mouse.x+100, game.mouse.y), # right
            (game.mouse.x-100, game.mouse.y), # left
            (game.mouse.x, game.mouse.y - 100), # up
            (game.mouse.x, game.mouse.y + 100), # down
        ]

        state = [
            # Cheese direction
            game.mouse.x < 700,
            game.mouse.x > 700,
            game.mouse.y < 700,
            game.mouse.y > 700,

            # Danger around
            aroundLocations[0] in data.catPositions,
            aroundLocations[1] in data.catPositions,
            aroundLocations[2] in data.catPositions,
            aroundLocations[3] in data.catPositions,   

            distanceToCheese 
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon =  data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(-data.decayRate * self.decayStep)
        final_move = [0,0,0,0]
        if np.random.rand() < self.epsilon:
            move = random.randint(0, 3)
            final_move[3] = 1
            self.randomMoves += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.aiMoves += 1
        self.decayStep += data.decayStep
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            aiMoves = agent.aiMoves
            randomMoves = agent.randomMoves
            totalMoves = aiMoves + randomMoves
            agent.aiMoves = 0
            agent.randomMoves = 0
            x,y = game.mouse.x, game.mouse.y
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Won', score, "Epsilon", agent.epsilon, "%", round(aiMoves/totalMoves * 100.0, 5), aiMoves, totalMoves, final_move, x, y)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

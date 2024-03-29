import torch
import random
import numpy as np
from collections import deque
from game import Game
from model import Linear_QNet, QTrainer
import data
from tqdm import trange

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, gamma, lr, maxMemory, hiddenSize):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=maxMemory)  # popleft()
        self.model = Linear_QNet(12, hiddenSize, 4)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)
        self.decayStep = 0
        self.aiMoves = 0
        self.randomMoves = 0
        self.lastMove = [0, 0, 0, 0]

    def get_state(self, game: Game):
        distanceToCheese = game.getDistanceToCheese()
        aroundLocations = [
            (game.mouse.x + 100, game.mouse.y),  # right
            (game.mouse.x - 100, game.mouse.y),  # left
            (game.mouse.x, game.mouse.y - 100),  # up
            (game.mouse.x, game.mouse.y + 100),  # down
        ]
        canMoveUp = game.mouse.y != 0
        canMoveDown = game.mouse.y != 700
        canMoveLeft = game.mouse.x != 0
        canMoveRight = game.mouse.x != 700
        lastMove = -1

        if self.lastMove[0] == 1:
            lastMove = 0
        elif self.lastMove[1] == 1:
            lastMove = 1
        elif self.lastMove[2] == 1:
            lastMove = 2
        elif self.lastMove[3] == 1:
            lastMove = 3

        state = [
            lastMove,
            canMoveUp,
            canMoveDown,
            canMoveLeft,
            canMoveRight,
            # Danger around
            aroundLocations[0] in data.catPositions,  # right
            aroundLocations[1] in data.catPositions,  # left
            aroundLocations[2] in data.catPositions,  # up
            aroundLocations[3] in data.catPositions,  # down
            distanceToCheese,
            game.mouse.x,
            game.mouse.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > data.batchSize:
            mini_sample = random.sample(self.memory, data.batchSize)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = data.epsilon - self.decayStep
        final_move = [0, 0, 0, 0]
        if random.randint(0, 10000) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
            self.randomMoves += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.aiMoves += 1
        self.decayStep += data.decayStep
        return final_move


def train(gamma, lr, maxMemory, hiddenSize, numberOfSteps, i):
    # won = False
    agent = Agent(gamma, lr, maxMemory, hiddenSize)
    game = Game()
    wonRound = False
    gameWhenWon = numberOfSteps
    for j in trange(
        # 22,
        numberOfSteps,
        desc=f"{i}: {gamma}, {lr}, {maxMemory}, {hiddenSize}".ljust(70, " "),
    ):
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        agent.lastMove = final_move
        # perform move and get new state
        reward, done, won = game.step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            agent.aiMoves = 0
            agent.randomMoves = 0
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if won:
                agent.model.save()
                wonRound = True
                gameWhenWon = agent.n_games
                return True, agent.n_games

            if agent.n_games % 2000 == 0:
                agent.trainer.lr = agent.trainer.lr / 2

    return wonRound, gameWhenWon


if __name__ == "__main__":
    gameList = []
    won, nGames = train(
        data.gamma, data.lr, data.maxMemory, data.hiddenSize, data.numberOfGames, -1
    )
    gameList.append(
        [
            won,
            nGames,
            data.gamma,
            data.lr,
            data.maxMemory,
            data.hiddenSize,
            data.numberOfGames,
        ]
    )

    def RNG():
        for i in range(100):
            gamma = random.random()
            lr = 10 / (10 ** random.randint(1, 7))
            maxMemory = int(random.uniform(10, 1_000_000))
            hiddenSize = 2 ** random.randint(2, 8)
            won, nGames = train(gamma, lr, maxMemory, hiddenSize, data.numberOfGames, i)
            gameList.append(
                [won, nGames, gamma, lr, maxMemory, hiddenSize, data.numberOfGames]
            )
        print(gameList)

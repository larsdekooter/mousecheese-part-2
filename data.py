catPositions = [
    (100, 100),
    (200, 100),
    (300, 100),
    (500, 100),
    (600, 100),
    (700, 100),
    (0, 200),
    (500, 200),
    (100, 300),
    (300, 300),
    (400, 300),
    (500, 300),
    (400, 400),
    (400, 500),
    (400, 600),
    (600, 600),
    (600, 700),
]
cooldown = 0
cheeseReward = 1000000
rewardNerf = 5
gamma = 0.9
maxMemory = 100_000
hiddenSize = 6  # Increased hidden layer size for more capacity
lr = 0.00001  # Adjusted learning rate for stability
maxEpsilon = 1
minEpsilon = 0.01
decayRate = 0.001
batchSize = 64  # Adjusted batch size for efficiency
random = 200
testLength = 6000
invalidMovePunishment = -10000
greaterThan = 0.1
decayStep = 1
epsilon = 10000


def getDistanceReward(distance):
    return 4 ** (10 - (distance / 100))  # Slightly increased reward decay


def getEfficiencyPenalty(distance):
    return 0.0005 * distance  # Reduced efficiency penalty for exploration


numberOfGames = 10000

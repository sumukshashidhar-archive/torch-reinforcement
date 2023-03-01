from collections import deque
import numpy as np
from model import LinearQNet, QTrainer
from wrapper import ResponseBitWrapper as Wrapper
import torch
import random
from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 10
LR = 0.1 # learning rate

class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = LinearQNet(1, 32, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    

    def get_state(self, game: Wrapper) -> int:
        # we want to return the challenge bit only as the state
        return np.array([game.game.get_value()], dtype=np.int32)


    def remember(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state: int) -> int:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = 0
        if random.randint(0, 200) < self.epsilon:
            final_move = random.randint(0, 1)
        else:
            current_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(current_state)
            final_move = 1 if float(prediction[0]) > 0.5 else 0
        print("State: ", state)
        print("Predicted Final Move: ", final_move)
        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    # initialize the agent
    agent = Agent()
    # initialize the game
    game = Wrapper()
    # initialize the score
    total_score = 0
    # now, let us play the game
    record = 0
    # in a loop
    while True:
        # first, let us get the current state of the game
        current_state = agent.get_state(game)
        # okay, this means that we have the challenge bit in this case
        # now, let us get the action
        action = agent.get_action(current_state)
        # now, let us play the step
        reward, done, score = game.play_step(action)
        # now, let us get the next state
        next_state = agent.get_state(game)
        # print("Reward: ", reward)
        # print("Done: ", done)
        # print("Score: ", score)
        # print("Next State: ", next_state)
        # input()
        # now, let us remember this
        agent.remember(current_state, action, reward, next_state, done)
        # now, let us train the short memory
        agent.train_short_memory(current_state, action, reward, next_state, done)
        # now, let us train the long memory
        if done:
            # this means that we lost the game
            # so, let us train the long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save('model.pth')

            print("Game", agent.n_games, "Score", score, "Record:", record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            



if __name__ == "__main__":
    train()
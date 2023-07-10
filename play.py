import gymnasium
import numpy as np
import torch 
from collections import deque
import random 
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean

#A simple NN
class q_NN(torch.nn.Module):
    def __init__(self, env):
        super(q_NN, self).__init__()
        self.input_shape = env.observation_space.shape[0]
        self.actions = env.action_space.n
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.actions)
        )

    def forward(self, x):
        return self.fc(torch.as_tensor(x))
    
env = gymnasium.make("LunarLander-v2", render_mode = "human")
def play():
  observation, _ = env.reset()
  while True:
    with torch.no_grad():
            action = np.argmax(NN(observation).numpy(), axis=0)
    new_observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    observation = new_observation
    if done: observation, info = env.reset()


Q = q_NN(env)
Q.load_state_dict(torch.load("your path.pt"))
play()

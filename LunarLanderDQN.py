import gymnasium
import numpy as np
import torch
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean
import wandb


###WANDB CONFIG
wandb.login()

#Hyperparameter search performed randomly
sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'c_reward',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric

#The parameters with their ranges
parameters_dict = {
    "steps" : {"value" : 300000},
    "sync": { "values" : [500, 1000, 5000, 10000]} ,
    "epsilon_final": {"value" : 0.01},
    "epsilon_initial": {"value" : 1},
    "epsilon_final_step": { "values" : [10000, 100000]} ,
    "memory_size": { "values" : [1000, 10000, 100000]} ,
    "replay_starting_size": { "values" : [1000, 10000]} ,
    "batch_size": { "value" : 128},
    "gamma": { "value" : 0.99},
    "lr" : {"values" : [0.001, 0.0005]}
    }

# steps = total_steps
# sync = n. steps between networks sync
# eps_initial, eps_final = epsilon initial and final values
# epsilon_final_step = final step of the decay
# memory_size = size of the replay buffer
# replay_starting_size = size the replay memory has to reach before sampling beings
# batch_s = batch size of the replay buffer
# gamma = discount ratec
# lr = learning rate

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="RL_Lander_DQN")

##Initializing the environment
env = gymnasium.make("LunarLander-v2")

###DQN Config
#Replay Memory
class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen = size)

    def insert(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

#epsilon decay
def decay(total_n, n, initial, final):
    '''
    total_n = total number of steps
    n = current step
    initial/final values of epsilon
    '''
    rate = (final - initial) / total_n
    return max(final, initial + rate *n)

#epsilon-greedy policy
def e_greedy(epsilon, nn, state):
    s = torch.as_tensor(state)
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return np.argmax(nn(state).numpy(), axis=0)

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


#Here the training loop is defined
def train_loop(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        #Initialize target and base networks and replay buffer
        mem = ReplayMemory(config.memory_size)
        Q = q_NN(env)
        Q_hat = q_NN(env)
        Q_hat.load_state_dict(Q.state_dict())
        optimizer = torch.optim.AdamW(Q.parameters(), lr=config.lr)
        MSE = torch.nn.MSELoss()
        #Starting state
        observation, _ = env.reset()

        #Some counters for tracking results
        ep_counter = 1
        c_reward = []
        ep_reward = 0
        ep_loss = 0
        c_loss = []

        for t in tqdm(range(config.steps)):
            #chose action accorfding to an epsililon greedy with decaying epsilon
            epsilon = (1 if t < config.replay_starting_size else decay(config.epsilon_final_step, t, config.epsilon_initial, config.epsilon_final))
            action = e_greedy(epsilon, nn = Q, state= observation)
            #perform action and save results in memory
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            mem.insert([observation, action, reward, new_observation, done])
            observation = new_observation
            #sample a random mini-batch of transitions from the replay buffer if it is fulled enough
            if t > config.replay_starting_size:
                #sampling from memory and converting to tensor
                batch = mem.sample(config.batch_size)
                st = torch.tensor(np.asanyarray([i[0] for i in batch])).float()
                a =  torch.tensor(np.asanyarray([i[1] for i in batch]))
                r =  torch.tensor(np.asanyarray([i[2] for i in batch])).float()
                st_1 =  torch.tensor(np.asanyarray([i[3] for i in batch])).float()
                d = torch.tensor(np.asanyarray([i[4] for i in batch])).int()
                #Calculate the target yi
                maxQ = Q_hat(st_1).max(1)[0]
                maxQ = maxQ.detach()
                yi = r + config.gamma * maxQ * (1- d)
                #Get the expected
                expected = torch.gather(Q(st), 1, a.unsqueeze(-1))
                #Calculate loss: ℒ = (𝑄𝑄(𝑠𝑠, 𝑎𝑎) − 𝑦𝑦)^2.
                loss = MSE(expected.squeeze(-1), yi)
                #Update Q(s, a) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            ep_reward = ep_reward + reward
            if done:
                #Update the tracking
                c_reward.append(ep_reward)
                c_loss.append(ep_loss)
                wandb.log({"loss": ep_loss, "episode_reward": ep_reward})
                ep_reward = 0
                ep_loss = 0
                if ep_counter%10 == 0:
                    print(f"The reward over the last ten episodes is {mean(c_reward[(ep_counter-10):ep_counter])}")
                    print(f"The loss over the last ten episodes is {mean(c_loss[(ep_counter-10):ep_counter])}")
                ep_counter += 1
                observation, info = env.reset()
            #Sync the two netwroks
            if t%config.sync == 0:
                Q_hat.load_state_dict(Q.state_dict())
        #Saving the weights
        rng = np.random.default_rng()
        save_name = f"landerDQN{rng.random()}.pt"
        torch.save(Q.state_dict(), save_name)

#Start a run
wandb.agent(sweep_id, train_loop, count=5)
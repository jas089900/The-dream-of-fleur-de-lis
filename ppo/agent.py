import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, log_std_init=0.0):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.mean = nn.Linear(32, action_size)
        self.log_std = nn.Parameter(torch.ones(1, action_size) * log_std_init)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.value = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)
        return value

class Agent:
    def __init__(self, state_size, action_size, filepath=None):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)

        if filepath is not None:
            self.load_model(filepath)

        self.old_policy_net = PolicyNetwork(state_size, action_size)
        self.update_old_policy()

    def update_old_policy(self):
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state): #没问题了
        state = torch.from_numpy(state).float()
        mean, std = self.policy_net(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach().numpy()[0], log_prob.detach().numpy()[0], self.value_net(state).detach().numpy()[0]

    def compute_gae(self, rewards, values, next_values, gamma=0.99, lam=0.97): #这个没问题
        deltas = [r + gamma * v_next - v for r, v_next, v in zip(rewards, next_values, values)]
        gaes = np.array(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + gamma * lam * gaes[t + 1]
        return gaes

    def train(self, states, actions, rewards, old_log_probs, advantages, log_stds): #这是个on-policy的版本
        #原来的项目好像是off-policy,就是不更新old_policy
        #目前的算法实际上是PPO-CLIP，没有加上熵损失，后面有时间再加
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(values, rewards)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        mean, std = self.policy_net(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 +0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.update_old_policy()

    def save_model(self, step, frequency=20000): #已没问题
        if (step % frequency == 0):
            filepath = './model/walking__{}'.format(step)
            torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
             }, filepath)
            print("Saved model to {}".format(filepath))
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCriticCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 120, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(512, 2)
        self.actor_log_std = nn.Parameter(torch.zeros(2))
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        features = self.fc(self.cnn(obs))
        mean = torch.tanh(self.actor_mean(features))
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value


class PPOAgent:
    def __init__(self):
        self.net = ActorCriticCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4)

        self.obs       = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []

    def choose_action(self, observation):
        img = observation["image"]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            mean, std, value = self.net(img)

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        self._last_log_prob = log_prob.item()
        self._last_value = value.item()

        action = action.squeeze(0).numpy()
        action[0] = float(np.clip(action[0], 0.0, 1.0))   # velocity
        action[1] = float(np.clip(action[1], -1.0, 1.0))  # steering
        return action.tolist()

    def store_transition(self, obs, action, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(self._last_log_prob)
        self.values.append(self._last_value)

    def learn(self):
        if len(self.rewards) < 2:
            self.obs, self.actions, self.rewards = [], [], []
            self.dones, self.log_probs, self.values = [], [], []
            return

        # Calcul des avantages (GAE)
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = 0.0 if self.dones[t] else (self.values[t+1] if t+1 < T else 0.0)
            delta = self.rewards[t] + 0.99 * next_value - self.values[t]
            last_gae = delta + 0.99 * 0.95 * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Prépare les tensors
        obs_tensors = torch.cat([
            torch.tensor(o["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            for o in self.obs
        ], dim=0)

        actions_t     = torch.tensor(self.actions,   dtype=torch.float32)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        advantages_t  = torch.tensor(advantages,     dtype=torch.float32)
        returns_t     = torch.tensor(returns,        dtype=torch.float32)

        # Mise à jour PPO
        for _ in range(4):
            mean, std, values = self.net(obs_tensors)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages_t
            surr2 = ratio.clamp(0.8, 1.2) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = nn.functional.mse_loss(values, returns_t)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Vide les listes
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path))
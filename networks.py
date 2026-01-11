import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, state, deterministic=False):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(dist.probs)
            log_prob = dist.log_prob(action).detach()  # 🔑 detach
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value
    
    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value
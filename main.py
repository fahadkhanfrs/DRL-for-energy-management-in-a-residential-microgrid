# from env import MicrogridEnv

# env = MicrogridEnv()
# state = env.reset()

# # for i in range(6):
# #     a = i # alternate charge/discharge
# #     s, r, d, _ = env.step(a)
# #     print(f"t={i+1}, SoC={s[1]:.3f}")

# # for _ in range(5):
# #     action = 10  # Example action
# #     next_state, reward, done, info = env.step(action)
# #     print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
# #     if done:
# #         break

# for i in range(6):
#     action = 0 if i < 3 else 79  # charge for 3 steps, then discharge for 3 steps
#     s, r, d, info = env.step(action)
#     print(
#         f"t={i+1} | ",
#         f"SoC={s[1]:.3f} | ",
#         f"Grid Import={info['grid_import']:.1f} | ",
#         f"Grid Export={info['grid_export']:.1f} | ",
#         f"Reward={info['reward']:.1f}"
#     )

    # ----------------------------------------------------------------------------

import torch
import numpy as np
from env import MicrogridEnv
from networks import ActorCritic
from ppo import compute_gae, ppo_update

env = MicrogridEnv()

state_dim = env.state_dim()
action_dim = 80

policy = ActorCritic(state_dim, action_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

EPISDODES = 750
episode_returns_ppopp = []
episode_entropies_ppopp = []

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 300

def epsilon_by_episode(ep):
    return max(
        EPS_END,
        EPS_START - ep / EPS_DECAY_EPISODES
    )


for episode in range(EPISDODES):
    state = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        eps = epsilon_by_episode(episode)
        deterministic = np.random.rand() > eps

        action, log_prob, value = policy.act(
            state_tensor,
            deterministic=deterministic
        )

        next_state, reward, done, _ = env.step(action.item())

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        state = next_state

    # Compute GAE and returns
    advantages, returns = compute_gae(
        rewards,
        [v.item() for v in values],
        dones
    )

    # Convert to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(log_probs).detach()
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO update
    stats = ppo_update(
        policy,
        optimizer,
        states,
        actions,
        old_log_probs,
        returns,
        advantages
    )
    
    if episode % 10 == 0:
     print(
            f"Episode {episode} | "
            f"Return {sum(rewards):.1f} | "
            f"Entropy {stats['entropy_bonus']:.3f}"
        )
     
     episode_returns_ppopp.append(sum(rewards))
     episode_entropies_ppopp.append(stats["entropy_bonus"])

     np.save("episode_returns_ppopp.npy", np.array(episode_returns_ppopp))
     np.save("episode_entropies_ppopp.npy", np.array(episode_entropies_ppopp))

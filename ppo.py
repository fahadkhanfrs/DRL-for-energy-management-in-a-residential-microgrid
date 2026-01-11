# PPO utilities (no changes made)
import torch
import torch.nn.functional as F

# GAE calculation

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

# PPO update step

def ppo_update(policy,
                optimizer,
                states,
                actions,
                old_log_probs,
                returns,
                advantages,
                clip_eps=0.2,
                value_loss_coef=0.5,
                entropy_coef=0.01
                ):
    log_probs, entropy, values = policy.evaluate(states, actions)

    ratios = torch.exp(log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -torch.min(
        ratios * advantages,
        clipped_ratios * advantages
        ).mean()
    
    value_loss = F.mse_loss(values, returns)

    entropy_bonus = entropy.mean()

    loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_bonus": entropy_bonus.item()
    }
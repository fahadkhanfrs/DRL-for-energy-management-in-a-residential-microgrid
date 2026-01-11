import numpy as np
import matplotlib.pyplot as plt

# Loading the data

returns_ppo = np.load('episode_returns_ppo.npy')
entropies_ppo = np.load('episode_entropies_ppo.npy')

returns_ppopp = np.load('episode_returns_ppopp.npy')
entropies_ppopp = np.load('episode_entropies_ppopp.npy')

episodes = np.arange(len(returns_ppo))

# --- Plotting Returns
# plt.figure()
# plt.plot(episodes, returns_ppo, label='PPO', color='blue')
# plt.plot(episodes, returns_ppopp, label='PPOPP', color='orange')
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.legend()
# plt.title('Episode Returns Comparison')
# plt.savefig("figures/return_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()

# # --- Plotting Entropies
# plt.figure()
# plt.plot(episodes, entropies_ppo, label='PPO', color='blue')
# plt.plot(episodes, entropies_ppopp, label='PPOPP', color='orange')
# plt.xlabel('Episode')
# plt.ylabel('Entropy')
# plt.legend()
# plt.title('Episode Entropy Comparison')
# plt.savefig("figures/entropy_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()

# def moving_average(data, window_size=10):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# last 100 episodes only
tail = 75

ppo_tail = returns_ppo[-tail:]
ppopp_tail = returns_ppopp[-tail:]

print("=== PPO (tail) ===")
print("Mean:", ppo_tail.mean())
print("Std :", ppo_tail.std())

print("\n=== PPO++ (tail) ===")
print("Mean:", ppopp_tail.mean())
print("Std :", ppopp_tail.std())


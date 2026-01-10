from env import MicrogridEnv

env = MicrogridEnv()
state = env.reset()

for _ in range(5):
    action = 10  # Example action
    next_state, reward, done, info = env.step(action)
    print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")
    if done:
        break
import numpy as np

class MicrogridEnv:
    def __init__(self):
        self.t = 0
        self.max_steps = 24  # hours

        # placeholders for state and action dimensions
        self.state = None

    def reset(self):
        self.t = 0
        # Initialize state with dummy values
        self.state = np.zeros(self.state_dim())
        return self.state
    
    def step(self, action):
        # Dummy implementation of step
        # action is an int in [0, 79]

        assert 0 <= action < 80, "Action out of bounds"

        self.t += 1

        # placeholder dynamics
        reward = 0.0
        done = self.t >= self.max_steps
        reward = -np.sum(action**2)  # Example reward function

        # state transition placeholder
        self.state = np.zeros(self.state_dim())

        info = {}
        return self.state, reward, done, info
    
    @staticmethod
    def state_dim():

        # explicitly define state dimension
        return 8
    
    """ 
        states_t = [avg_TCL_SoC, ESS_Soc,
            price_counter, outdoor_temp,
            wind_generation,
            up_regulation_price, base_load, hour_of_day]
    """
    

import numpy as np

class MicrogridEnv:
    def __init__(self):
        self.t = 0
        self.max_steps = 24  # hours

        # ESS parameters
        self.ESS_capacity = 500.0  # kWh
        self.eta_charge = 0.95
        self.eta_discharge = 0.95
        self.max_charge = 100.0  # kW
        self.min_discharge = -100.0  # kW

        self.ESS = None # Current stored energy in ESS

        # Grid parameters
        self.grid_price_buy = 5.0 # arbitrary constant
        self.grid_price_sell = 4.0 # arbitrary constant

        self.grid_import = 0.0 # energy bought this hour
        self.grid_export = 0.0 # energy sold this hour

        # Dummy loads and generation
        self.fixed_load = 200.0  # kW per hour
        self.fixed_generation = 150.0

        # placeholders for state and action dimensions
        self.state = None

    def reset(self):
        self.t = 0

        # Start at half capacity
        self.ESS = self.ESS_capacity / 2.0

        # Initialize state with dummy values

        self.state = np.zeros(self.state_dim())
        return self.state
    
    def _get_state(self):
        state = np.zeros(self.state_dim())

        # state[1] = ESS SoC
        state[1] = self.ESS / self.ESS_capacity

        # state[7] = hour of day
        state[7] = self.t

        state[5] = self.grid_price_buy  # up_regulation_price

        return state

    
    def step(self, action):
        # Dummy implementation of step
        # action is an int in [0, 79]

        assert 0 <= action < 80, "Action out of bounds"

        self.t += 1

        # Dummy ESS control (temporary placeholder logic)

        if action < 40:
            charge_power = (action / 39.0) * self.max_charge
            self.ESS += charge_power * self.eta_charge
        else:
            discharge_power = ((action - 40) / 39.0) * abs(self.min_discharge)
            self.ESS -= discharge_power / self.eta_discharge

        # Enforce physical bounds
        self.ESS = np.clip(self.ESS, 0, self.ESS_capacity)

        # Net energy balance
        net_energy = self.fixed_generation - self.fixed_load

        # Adjust for ESS actions
        if action < 40:
            net_energy -= charge_power
        else:
            net_energy += discharge_power

            # Grid balances the rest
        if net_energy >= 0:
            # surplus energy sold to grid
            self.grid_export = net_energy
            self.grid_import = 0.0
        else:
            # deficit energy bought from grid
            self.grid_import = -net_energy
            self.grid_export = 0.0

        # placeholder dynamics
        reward = (
          self.grid_export * self.grid_price_sell
          - self.grid_import * self.grid_price_buy
        )

#         print(
#     f"[DEBUG] import={self.grid_import}, "
#     f"export={self.grid_export}, "
#     f"reward={reward}"
# )


        done = self.t >= self.max_steps
        # state transition placeholder
        # self.state = np.zeros(self.state_dim())
        self.state = self._get_state()

        info = {
            "grid_import": self.grid_import,
            "grid_export": self.grid_export,
            "battery_soc": self.ESS,
            "reward": reward
        }
        
        return self.state, reward, done, info
    
    @staticmethod
    def state_dim():

        # explicitly define state dimension
        return 8
    
    """"
    states_t = [avg_TCL_SoC, ESS_Soc, price_counter, outdoor_temp, wind_generation, up_regulation_price, base_load, hour_of_day]
    """
    

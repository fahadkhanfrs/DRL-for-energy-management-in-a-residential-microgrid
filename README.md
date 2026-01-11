# DRL-for-energy-management-in-a-residential-microgrid

This microgrid has multiple sources of flexibility on both the supply side and the demand side.

It combines:

Wind generation (uncertain supply),

A shared battery (ESS),

Thermostatically Controlled Loads (TCLs) → direct control,

Price-responsive residential loads → indirect control via pricing,

Connection to the main grid.

I have designed an Energy Management System (EMS) that coordinates all of the above.

Later, this is cast as an MDP and solved with DRL.

Compared to vanilla PPO, PPO++ exhibits significantly improved training stability and higher average episodic return. PPO frequently suffers from policy regression due to excessive stochastic exploration, whereas PPO++ maintains profitable behaviors once discovered.

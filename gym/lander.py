import gymnasium as gym
env = gym.make(
    "LunarLander-v2",

    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
)

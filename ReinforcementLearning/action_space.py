"""
Only Gymnasium supports setting a start parameter in Discrete(int, start=int), unlike Gym.

option 1
bale_dim: (resize_direction, resize_factor, laft, lfore)
transom: [transom_widt, transom_height] # transom_offset doesn't make a lot of sense

option 2
bale_dim: (lhold, B/L, laft, lfore)
transom: [transom_widt, transom_height] # transom_offset doesn't make a lot of sense
The problem I'm facing now is that the action space is dependent on the hull shape, but the hull shape is dependent on the action space. 
"""

from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete, Box
from gym.spaces.utils import flatten_space
from gym import spaces
import numpy as np
from scipy.interpolate import interp1d
action_space = Dict(
        {
        'bale_dim': Tuple((Discrete(250), Discrete(6), Discrete(9), Discrete(9))),
        'transom': MultiDiscrete([20, 20]),
        'bilge_radius': Discrete(4),
        'ctrlpts_offset': MultiDiscrete([50])
    }        
)
action = MultiDiscrete([250, 6, 9, 9, 20, 20, 4, 50])
action2 = Box(low=np.array([1, 4, 3, 3, 0, 0, 0, 0]), high=np.array([250, 6, 9, 9, 20, 20, 4, 25]), shape=(8,), dtype=np.float64)
normalized = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1, 1]), shape=(8,), dtype=np.float64)

print(action2)
actor = normalized.sample()[1]

x = np.linspace(1, 250, 50)
y = np.linspace(0, 1, 50)
f = interp1d(x, y)
print(actor)
print(f(actor))
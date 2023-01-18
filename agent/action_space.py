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

from gymnasium.spaces import Dict, Discrete, Tuple, MultiDiscrete

action_space = Dict(
        {
        'bale_dim': Tuple((Discrete(250, start=1),Discrete(6, start=4), Discrete(9, start=3), Discrete(9, start=3))),
        'transom': MultiDiscrete([20, 20]),
        'bilge_radius': Discrete(4),
        'ctrlpts_offset': MultiDiscrete([50])
    }        
)





action_space2 = Dict({'bale_dim': MultiDiscrete([7, 5])})

r = Discrete(6, start=4)
r.seed(4)
# print(action_space.sample())
# print(action_space.sample()['transom'][0])
for _ in range(20):
    print(r.sample())
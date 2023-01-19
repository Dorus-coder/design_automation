"""
The goal of the agent is to alter the ship until the lowest resistance is found.

observations: Observations provide the current state of the system. Or in other words, the geometry of the hull.

In the current enviroment the shape is mostly defined by the waterplane, the web frame and the transom.
Reward: The agent recieves a reward of 1 if the resistance decreases and a reward of -1 if the resistance increases.
done: The enviroment is done when there is no design with a lower resistance found.

Sources:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations
    https://stable-baselines3.readthedocs.io/en/master/guide/install.html
    https://gymnasium.farama.org/tutorials/environment_creation/
"""
from build_vessel.parameters import Block, CtrlPts
import gym
import numpy as np
from gym import spaces

BALE = 8000

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, arg1, arg2):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        """
        Arg:
            action (dict): During training these are random samples, after training it is a prediction from the model.
        Return:
            Observation: A numpy array with the input of the parametric model.

        """
        # The observation of the enviroment consist the 

        # this function should check if the offset parameters are not bigger than the length in x direction of the B-spline.
        # and also for the transom
        self.md.action = action
        self.md.maindim()
        self.block = Block(laft=self.md.laft, 
                      lhold=self.md.lhold,
                      lfore=self.md.lfore,
                      boa=self.md.boa,
                      depth=self.md.depth_hold,
                      bilge_radius=action['bilge_radius'],
                      ctrlpt_offset_forward=action["ctrlpts_offset"][0],
                      transom_width=action['transom'][0],
                      transom_height=action['transom'][1],
                    )
        cp = CtrlPts(block)
        return observation, reward, done, info

    def reset(self):
        """
        This function should delete the parametric model obj and del the control points
        """
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        """
        This function initialize the parametric model.
        Therefore it should set the parameters
        """
        from build_vessel.parameters import MainDimGenerator
        self.md = MainDimGenerator(bale=BALE)

    def close(self):
        ...

    def make_frames(self, ctrlpts: CtrlPts):
        from build_vessel.cross_section import CrossSection
        from build_vessel.waterplane import WaterPlane
        from build_vessel.helpers import modify_control_points
        transom = CrossSection(ctrlpts.cross_frames.transom)
        wbfrm = CrossSection(ctrlpts.cross_frames.web_frame)
        fpp = CrossSection(ctrlpts.cross_frames.fpp_frame)
        # control points at the aft of the hold based on the web frame
        hold_aft_ctrlpts = modify_control_points(ctrlpts.web_frame, 0 , self.block.laft)
        hold_aft = CrossSection(hold_aft_ctrlpts)

        wp = WaterPlane(ctrlpts.waterlines.waterplane)


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
from build_vessel.cross_section import CrossSection, BuildFrames
from build_vessel.properties import Properties, Info
from build_vessel.utils import modify_control_points, HMInput

BALE = 8000
VELOCITY = 12


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.hm_resistance = []
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=0, high=255,
                                            # shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

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
        cp = CtrlPts(self.block)
        done = cp.check_ctrlpts(action)
        
        info, observation = self.observe_resistance(cp)
        info.laft = self.block.laft
        info.lhold = self.block.lhold
        info.lfore = self.block.lfore
        info.lwl = sum((info.laft, info.lhold, info.lfore))
        info.half_boa = self.block.boa
        info.draft = self.block.draft
        
        reward = self.reward_function()
        return observation, reward, done, info #observation, reward, done, info

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
        Maybe render should be the visulization of the ship. Resetting the parameters of the ship every iteration of
        the episode might not be a good thing. 
        
        """
        from build_vessel.parameters import MainDimGenerator
        self.md = MainDimGenerator(bale=BALE)

    def close(self):
        """
        This function should close the visualization of the ship.
        """
        return
    def main_frames(self, ctrlpts: CtrlPts):
        from build_vessel.waterplane import WaterPlane
        

        transom = CrossSection(ctrlpts.cross_frames.transom)
        self.wbfrm = CrossSection(ctrlpts.cross_frames.web_frame)
        fpp = CrossSection(ctrlpts.cross_frames.fpp_frame)
        
        # control points at the aft of the hold based on the web frame
        self.hold_aft_ctrlpts = modify_control_points(ctrlpts.web_frame, 0 , self.block.laft)
        hold_aft = CrossSection(self.hold_aft_ctrlpts)
        hold_fore_ctrlpts = modify_control_points(ctrlpts.web_frame, 0, (self.block.laft + self.block.lhold))
        hold_fore = CrossSection(hold_fore_ctrlpts)
        self.wp = WaterPlane(ctrlpts.waterlines.waterplane)
        return transom, fpp, hold_aft, hold_fore_ctrlpts, hold_fore

    def frames(self, ctrlpts: CtrlPts):
        transom, fpp, hold_aft, hold_fore_ctrlpts, hold_fore = self.main_frames(ctrlpts)
        self.bf = BuildFrames(self.wp, self.block.laft, self.block.draft)
        aft = self.bf.aft(self.block.laft ,self.hold_aft_ctrlpts, ctrlpts.transom)
        mid = self.bf.midship(hold_aft.points, hold_fore.points)
        fore =self.bf.forward(hold_fore.points)
        return np.concatenate((aft, mid, fore), axis=0)

    def observe_resistance(self, ctrlpts: CtrlPts):
        from HoltropMennen import HoltropMennen
        points = self.frames(ctrlpts)
        info = Info()
        prop = Properties(self.block.draft, len(points), info)
        prop.memory = points, True
        prop.area()
        hm_input = HMInput(lpp= self.block.lwl,
                    B= self.block.boa * 2,
                    t_f= self.block.draft,
                    t_a= self.block.draft,
                    displ= prop.volume_scipy() * 2 * 1.025,
                    lcb=prop.lcb(),
                    c_m=self.wbfrm.cross_section_coefficient(),
                    c_wp=self.wp.c_wp(self.block.lwl, self.block.boa),
                    c_b=prop.block_coefficient(self.block.lwl, self.block.boa, self.block.draft),
                    a_t=prop.transom_area,
                    c_prism= prop.prismatic_coefficient(self.wbfrm.area, self.block.lwl),
                    ie=prop.ie(self.block.boa, self.block.lfore), 
                    velocity=VELOCITY,                
                    )
        hm_res = HoltropMennen(hm_input)
        hm_total_res = hm_res.total_resistance()
        self.hm_resistance.append(hm_total_res)
        return info, hm_total_res

    def reward_function(self):
        if len(self.hm_resistance) > 1:
            if self.hm_resistance[-1] < np.min(self.hm_resistance[:-1]):
                return 1
            else:
                return -1
        else:
            return 0


if __name__ == "__main__":
    from action_space import action_space, example
    env = CustomEnv()
    episodes = 10
    for episode in range(1, episodes+1):
        done = False
        score = 0
        while not done:
            env.render()
            # env.reset()
            sam = action_space.sample()
            observation, reward, done, info = env.step(sam)
            
            score += reward
            print(f"{reward = }")
            print(f"{sam = }")
        
            info.print_info()
            print(observation)
            print(f"end of iteration {done}")
            # env.bf.visualize()
            # env.bf.close_visualisation()
            
        print("\n \n \n")
        print(f"Episode: {episode} Score: {score}")
            # env.close()
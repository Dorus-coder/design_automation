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
from gym.spaces import Box
from build_vessel.utils import modify_control_points
from build_vessel.properties import Properties, Info
from build_vessel.cross_section import CrossSection, BuildFrames
from build_vessel.parameters import Block, CtrlPts, HMInput
import gym
import numpy as np
# np.seterr(invalid='raise')


BALE = 8000
VELOCITY = 12


class ShipEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.time_step = 0
        self.hm_resistance = np.array([np.inf])

        self.action_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]), high=np.array(
            [1, 1, 1, 1, 1, 1, 1, 1]), shape=(8,), dtype=np.float64)
        self.observation_space = Box(low=np.array(
            [0]), high=np.array([np.inf]), shape=(1,), dtype=np.float64)

        # self.action_space = Box(low=np.array([1, 4, 3, 3, 0, 0, 0, 0]), high=np.array([100, 6, 9, 9, 20, 20, 4, 25]), shape=(8,), dtype=np.float64)
        # self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1, 1]), shape=(8,), dtype=np.float64)

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
        self.time_step += 1

        from build_vessel.parameters import MainDimGenerator
        md = MainDimGenerator(bale=BALE)
        if np.nan in action:
            print(action)
            action = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        md.action = action
        md.maindim()
 
        self.block = Block(laft=md.laft,
                           lhold=md.lhold,
                           lfore=md.lfore,
                           boa=md.boa,
                           depth=md.depth_hold,
                           bilge_radius=action[6] * 4,
                           ctrlpt_offset_forward=action[7] * md.lfore,
                           transom_width=action[4] * md.boa,
                           transom_height_action=action[5]*md.depth_hold,
                           )
        cp = CtrlPts(self.block)
        done1 = self.block.check_done(action)
        done2 = False
        if self.time_step >= 100:
            done2 = True

        
        info, input_reward, observation, done3 = self.observe_resistance(cp)
        info.laft = self.block.laft
        info.lhold = self.block.lhold
        info.lfore = self.block.lfore
        info.lwl = self.block.lwl
        info.half_boa = self.block.boa
        info.draft = self.block.draft
        # try:
        #     info, observation = self.observe_resistance(cp)
        #     info.laft = self.block.laft
        #     info.lhold = self.block.lhold
        #     info.lfore = self.block.lfore
        #     info.lwl = self.block.lwl
        #     info.half_boa = self.block.boa
        #     info.draft = self.block.draft
        # except:
        #     ValueError("The resistance could not be calculated")
        #     return np.array([np.inf]), self.reward_function(), True, {"info": "ValueError raised: The resistance could not be calculated", "satus": "Failed!!!"}
            
        reward = self.reward_function(input_reward)
        done = False
        if True in [done1, done2, done3]:
            done = True
        return observation, reward, done, info.__dict__  # observation, reward, done, info

    def reset(self):
        # reset the reward
        observation = self.hm_resistance
        if len(self.hm_resistance) > 0:
            observation = np.array(min(self.hm_resistance))
        self.hm_resistance = np.array([np.inf])
        return np.array([observation])  # reward, done, info can't be included

    def render(self, mode="human"):
        """
        This function initialize the parametric model.
        Therefore it should set the parameters
        Maybe render should be the visulization of the ship. Resetting the parameters of the ship every iteration of
        the episode might not be a good thing. 

        """
        ...

    def close(self):
        """
        This function is probably necassary to stop the learning proces
        This function should close the visualization of the ship.
        """
        pass

    def main_frames(self, ctrlpts: CtrlPts):
        from build_vessel.waterplane import WaterPlane

        transom = CrossSection(ctrlpts.cross_frames.transom)
        self.wbfrm = CrossSection(ctrlpts.cross_frames.web_frame)
        fpp = CrossSection(ctrlpts.cross_frames.fpp_frame)

        # control points at the aft of the hold based on the web frame
        self.hold_aft_ctrlpts = modify_control_points(
            ctrlpts.web_frame, 0, self.block.laft)
        hold_aft = CrossSection(self.hold_aft_ctrlpts)
        hold_fore_ctrlpts = modify_control_points(
            ctrlpts.web_frame, 0, (self.block.laft + self.block.lhold))
        hold_fore = CrossSection(hold_fore_ctrlpts)
        self.wp = WaterPlane(ctrlpts.waterlines.waterplane)
        return transom, fpp, hold_aft, hold_fore_ctrlpts, hold_fore

    def frames(self, ctrlpts: CtrlPts):
        transom, fpp, hold_aft, hold_fore_ctrlpts, hold_fore = self.main_frames(
            ctrlpts)
        self.bf = BuildFrames(self.wp, self.block.laft, self.block.draft)
        aft = self.bf.aft(
            self.block.laft, self.hold_aft_ctrlpts, ctrlpts.transom)
        mid = self.bf.midship(hold_aft.points, hold_fore.points)
        fore = self.bf.forward(hold_fore.points)
        return np.concatenate((aft, mid, fore), axis=0)

    def observe_resistance(self, ctrlpts: CtrlPts):
        """
        Return:
            observation: np.array
            info: Info object
            done: Bool
        """
        from HoltropMennen import HoltropMennen
        
        points = self.frames(ctrlpts)
        
        info = Info()
        info.c_m = self.wbfrm.cross_section_coefficient()
        info.c_wp = self.wp.c_wp(self.block.lwl, self.block.boa)

        prop = Properties(self.block.draft, len(points), info)
        prop.memory = points, True
        prop.area()
        try:
            hm_input = HMInput(lpp=self.block.lwl,
                            B=self.block.boa * 2,
                            t_f=self.block.draft,
                            t_a=self.block.draft,
                            displ=prop.volume_scipy() * 2 * 1.025,
                            lcb=prop.lcb_ratio(self.block.lwl),
                            c_m=self.wbfrm.cross_section_coefficient(),
                            c_wp=self.wp.c_wp(self.block.lwl, self.block.boa),
                            c_b=prop.block_coefficient(
                                self.block.lwl, self.block.boa, self.block.draft),
                            a_t=prop.transom_area,
                            c_prism=prop.prismatic_coefficient(
                                self.wbfrm.area, self.block.lwl),
                            ie=prop.ie(self.block.boa, self.block.lfore),
                            velocity=VELOCITY,
                            )
        except ValueError:
            info.error = {"ValueError": "unkown error", 'state': np.inf}
            
            return info, "", np.array([np.inf]), True

        self.hm_res = HoltropMennen(hm_input)
        hm_total_res = self.hm_res.total_resistance()
        print(f"{hm_total_res = :.2f}")
        self.hm_resistance = np.append(self.hm_resistance, hm_total_res)
        return info, hm_input.reward_correct_input, np.array([hm_total_res]), False

    def reward_function(self, input_reward):
        reward = input_reward
        if len(self.hm_resistance) > 1:
            if self.hm_resistance[-1] < 0:
                reward += -2
            if self.hm_resistance[-1] < np.min(self.hm_resistance[:-1]):
                reward += 1
            else:
                return -1
        return reward


def main_1():
    env = ShipEnv()
    episodes = 20
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        num_iter = 0
        while not done:
            env.render()
            num_iter += 1
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            score += reward
            print("......................................................")
            print(f"{reward = }, {env.hm_resistance = }")
            print(f"{action = }")
            print()
            print(info)
            print(observation)
            print(f"end of iteration {done}")
            env.bf.visualize()
            # env.bf.close_visualisation()
            print("______________________________________________________")
        print("\n \n \n")
        print(f"Episode: {episode} Score: {score} Num_iter: {num_iter}")
        # env.close()

def main_2():
    from build_vessel.parameters import Block
    
    env = ShipEnv()
    
    env.block = Block(laft=30,
                  lhold=150,
                  lfore=20,
                  boa=16,
                  depth=13,
                  bilge_radius=2,
                  ctrlpt_offset_forward=7,
                  transom_width=14,
                  transom_height_action=0.8)
    cp = CtrlPts(env.block)
    points = env.frames(cp)
    info, reward, state, done = env.observe_resistance(cp)
    
    print(f"resistance: {state}")    
    for key, value in info.__dict__.items():
        print(f"{key}: {value}")

    env.bf.visualize()

if __name__ == "__main__":
    main_2()
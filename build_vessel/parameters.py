from dataclasses import dataclass, field
from build_vessel.freeboard import min_freeboard
import pandas as pd
import numpy as np
from enum import Enum
import copy

class Direction(Enum):
    """
    enumerates the resizing  direction.
    This is part of the action space 
    """
    XX = 0
    XY = 1 
    XZ = 2
    YZ = 3
    YX = 4
    ZX = 5
    ZY = 6


class MainDimGenerator:
    """
    This class generates main dimension based on the bale space. 
    """
    def __init__(self, bale, draft_limit= None, loa_limit= None, boa_limit= None) -> None:
        self.bale = bale
        self.draft_limit = draft_limit
        self.loa_limit = loa_limit
        self.boa_limit = boa_limit
        self._action = []  # input from the model or sample
        
        xyz = self.bale ** (1/3)
        self.lhold, self.boa, self.depth_hold = xyz, xyz, xyz

    @property
    def action(self):
        return self._action
    
    @action.setter
    def action(self, action_value):
        self._action = action_value

    def resize_bale(self):
        """
        scale the bale space in two dimensions\n
        goal is to find the best l/b \n
        Arg:
            direction (int) : input from the action space and is the resizing direction.
            factor (int): input from the action space and is the resizing factor.
        Return:
            x, y, z (tuple[float]): x= new length, y= new breadth, z= new hold
        """
        _direction = Direction(self.action[0])
        factor = self.action[1]
        scale_directions = {'XY': (1 / factor, factor, 1), 'XZ': (1 / factor, 1, factor), 'YZ': (1, 1 / factor, factor), 'YX': (factor, 1 / factor, 1), 'ZX': (factor, 1, 1 / factor), 'ZY': (1, factor, 1 / factor)}
        try:
            self.lhold *= scale_directions.get(_direction)[0]
            self.boa *= scale_directions.get(_direction)[1]
            self.depth_hold *= scale_directions.get(_direction)[2]
        except TypeError:
            # Do nothing if action is 0.
            return

    def maindim(self):

        self.lhold = self.action[0] * 100 + 25
        self.boa = self.lhold / (self.action[1] * 2 + 5)
        self.depth_hold = self.bale / (self.lhold * self.boa) 
        
    @property
    def laft(self):
        return int(np.ceil(self.lhold * self.action[2]))
        
    @property
    def lfore(self):
        return int(np.ceil(self.lhold * self.action[3]))


@dataclass
class Block:
    """Parameters describing the block of the hull shape.
    """
    laft: int
    lhold: float
    lfore: int
    boa: float
    lwl: float = field(init=False)
    loa: float = field(init=False)
    # draft should be the height of the hold minus the minimum freeboard
    depth: float
    draft: float = field(init=False)
    bilge_radius: int
    ctrlpt_offset_forward: int # Cannot become bigger than the length fore, because otherwise strange shapes emerge 
    # transom
    transom_width: float# reduction in breadth at the transom from the main deck
    transom_height_action: float
    transom_offset:int = 0
    # default values

    def __post_init__(self):
        self.lwl = sum((self.laft, self.lhold, self.lfore))
        self.loa = self.lwl
        self.draft = self.depth - min_freeboard(self.loa)
        self.transom_height = self.transom_height_action * self.draft
    
    def check_done(self, action):
        """Check if the actions are valid. If not, return True.\n
        The action space has to be made more maintable.
        """
        # check if transom width is bigger than the breadth
        if action[4] * self.boa > self.boa:
            return True
        # check if the transom height is bigger than the draft
        elif action[5] * self.draft > self.draft:
            return True
        # check if the control point offset is bigger than the length fore
        elif action[7] * self.lfore > self.lfore:
            return True
        # check if the draft is below zero.
        elif self.draft < 0:
            self.draft = 20 # in the case of a small draft, the other parameters are likely big and by increasing the draft unrealistically it cause an really big resistance.
            return True
        else:
            return False

@dataclass
class HMInput:
    lpp: float
    B: float
    t_f: float # draft fore
    t_a: float # draft aft
    displ: float
    lcb: float # Longitudinal center of bouyancy in percentage forward of 1/2 lpp
    c_m: float # midship section coefficient 
    c_wp: float # waterplane area coefficient
    a_t: float # transom area
    c_prism: float # prismatic coefficient 
    c_b: float
    ie: float # half angle of entrance 
    velocity: float
    c_stern: int = 0 # stern shape parameter
    # bulb optional
    h_b: float = 0.0001 # centre of bulb area above keel [m]
    a_bt: float = 0.001 # transverse bulb area [m^2]
    h_b: float = 0.0001 # centre of bulb area above keel [m]
    a_bt: float = 0.0001 # transverse bulb area [m^2]
    reward_correct_input: int = 0
    
    def __post_init__(self):
        if self.t_f < 0:
            self.t_a = 50
            self.t_f = 50
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            #raise ValueError("Field 't_a' cannot be negative.")
        if self.a_t < 0:
            self.a_t = 50
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            #raise ValueError("Field 'a_t' cannot be negative.")
        if self.displ < 0:
            self.displ = 1000000
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            #raise ValueError("Field 'displ' cannot be negative.")
        if not 0 > self.c_m > 1:
            self.c_m = 0.9
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            # raise ValueError("Field 'c_m' cannot be negative.")
        if not 0 > self.c_wp > 1:
            self.c_wp = 0.9
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            # raise ValueError("coefficients should be between 0 and 1")
        if not 0 > self.c_prism > 1:
            self.c_prism = 0.9
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            # raise ValueError("Field 'c_prism' cannot be negative.")
        if not 0 > self.c_b > 1:
            self.reward_correct_input -= -1
            self.c_b = 0.9
        else:
            self.reward_correct_input += 1
            # raise ValueError("Field 'c_b' cannot be negative.")
        if self.lcb < 0:
            self.lcb = 0
            self.reward_correct_input -= -1
        else:
            self.reward_correct_input += 1
            # raise ValueError("Field 'lcb' cannot be negative.")
            

# @dataclass  
# class Bulb:
#     length = block.loa - block.lwl
#     breadth = 3 # 1/2 breadth
#     bulb_offset = 0
#     HEIGHT = block.draft - bulb_offset


@dataclass
class Waterlines:
    waterplane : list
    main_deck : list

@dataclass
class CrossSectionFrames:
    web_frame : list
    transom : list
    fpp_frame : list


@dataclass
class Longitudinals:
    center : list


class CtrlPts:
    def __init__(self, block) -> None:
        self.block = block

        self.web_frame = [[self.block.loa / 2, 0, 0],
                    [self.block.loa / 2, self.block.boa - self.block.bilge_radius, 0],
                    [self.block.loa / 2, self.block.boa, 0],
                    [self.block.loa / 2, self.block.boa, self.block.bilge_radius],
                    [self.block.loa / 2, self.block.boa, self.block.draft]]

        self.main_deck = [[0, 0, self.block.depth],
                    [0, self.block.transom_width, self.block.depth],  
                    [self.block.laft, self.block.boa, self.block.depth], 
                    [self.block.laft + self.block.lhold, self.block.boa, self.block.depth], 
                    [self.block.loa, self.block.boa, self.block.depth], 
                    [self.block.loa, 0, self.block.depth]]

        # Change 11.48 with function value from finding where the transom intersects with the waterline.
        self.waterplane = [[0, 11.48, self.block.draft],
                    [0, self.block.boa, self.block.draft],
                    [self.block.laft, self.block.boa, self.block.draft],
                    [self.block.laft + self.block.lhold, self.block.boa, self.block.draft],
                    [self.block.lwl - self.block.ctrlpt_offset_forward, self.block.boa, self.block.draft],
                    [self.block.lwl, 0, self.block.draft]]

        self.transom = [[0, 0, self.block.transom_height],
                [0, self.block.transom_width, self.block.transom_height - self.block.transom_offset],
                [0, self.block.transom_width, self.block.depth]]

        # This is a special list with three sublists containing the first order lines aft, the bspline control points of the bulb and the first order lines fore
        bulb_long = [[self.block.lwl, 0, 0],
                    [self.block.loa, 0, 0],
                    [self.block.loa, 0, self.block.draft], 
                    [self.block.lwl, 0, self.block.draft]]

        mid_forward = [[self.block.lwl, 0, self.block.draft],
                    [self.block.loa, 0, self.block.depth]]

        self.longitudinal_center = [[[0, 0, self.block.depth],
                                [0, 0, self.block.transom_height],
                                [self.block.laft, 0, 0],
                                [self.block.lwl, 0, 0]],
                                bulb_long,
                                mid_forward]

        # cross section @ forward perpencidular
        self.frame_fpp = [bulb_long[0],
                    [self.block.lwl, 3, 0],
                    [self.block.lwl, 3, self.block.draft],
                    bulb_long[3]]
    @property    
    def cross_frames(self):
        return CrossSectionFrames(web_frame=self.web_frame, transom=self.transom, fpp_frame=self.frame_fpp)

    @property
    def longitudinals(self):
        return Longitudinals(self.longitudinal_center)

    @property
    def waterlines(self):
        return Waterlines(self.waterplane, self.main_deck)


if __name__ == '__main__':
    block = Block(laft=20, lhold=40, lfore=30, boa=20, depth=10, bilge_radius=2, ctrlpt_offset_forward=5, transom_width=5, transom_height=10)
    print(block)
    print(block.check_done([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
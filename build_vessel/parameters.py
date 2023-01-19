

from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum 
df = pd.read_csv(r"build_vessel\\Rules\\min_freeboard.csv")


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
        self.lhold = self.action['bale_dim'][0]
        self.boa = self.lhold / self.action[1]
        self.depth_hold = self.bale / (self.l_hold * self.boa) 

    @property
    def laft(self):
        return round(self.lhold / self.action[2])

    @property
    def lfore(self):
        return round(self.x / self.action[3])

    @property
    def loa(self):
        return np.sum((self.lfore, self.x, self.laft))

@dataclass
class Block:
    """Parameters describing the block of the hull shape.
    """
    laft: int = 25
    lhold: float = 145.0
    lfore: int = 30
    boa: float = 16.0
    # draft should be the height of the hold minus the minimum freeboard
    depth: float = 13
    bilge_radius: int = 2
    ctrlpt_offset_forward: int = 7 # Cannot become bigger than the length fore, because otherwise strange shapes emerge 
    # transom
    transom_width: int = 14 # reduction in breadth at the transom from the main deck
    height_transom: int = 8.5
    transom_offset:int = 0
    # default values
    lwl = laft + lhold + laft
    loa = lwl
    draft = depth - np.ceil(df.loc[df["length"] == loa].values[0][1] / 1000) # "A" type ship
    



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


# This would be better placed in a json file 
class CtrlPts:
    def __init__(self, block) -> None:

        self.web_frame = [[block.loa / 2, 0, 0],
                    [block.loa / 2, block.boa - block.bilge_radius, 0],
                    [block.loa / 2, block.boa, 0],
                    [block.loa / 2, block.boa, block.bilge_radius],
                    [block.loa / 2, block.boa, block.draft]]

        self.main_deck = [[0, 0, block.depth],
                    [0, block.transom_width, block.depth],  
                    [block.laft, block.boa, block.depth], 
                    [block.laft + block.lhold, block.boa, block.depth], 
                    [block.loa, block.boa, block.depth], 
                    [block.loa, 0, block.depth]]

        # Change 11.48 with function value from finding where the transom intersects with the waterline.
        self.waterplane = [[0, 11.48, block.draft],
                    [0, block.boa, block.draft],
                    [block.laft, block.boa, block.draft],
                    [block.laft + block.lhold, block.boa, block.draft],
                    [block.lwl - block.ctrlpt_offset_forward, block.boa, block.draft],
                    [block.lwl, 0, block.draft]]

        self.transom = [[0, 0, block.height_transom],
                [0, block.transom_width, block.height_transom - block.transom_offset],
                [0, block.transom_width, block.depth]]

            # This is a special list with three sublists containing the first order lines aft, the bspline control points of the bulb and the first order lines fore
        bulb_long = [[block.lwl, 0, 0],
                    [block.loa, 0, 0],
                    [block.loa, 0, block.draft], 
                    [block.lwl, 0, block.draft]]

        mid_forward = [[block.lwl, 0, block.draft],
                    [block.loa, 0, block.depth]]

        self.longitudinal_center = [[[0, 0, block.depth],
                                [0, 0, block.height_transom],
                                [block.laft, 0, 0],
                                [block.lwl, 0, 0]],
                                bulb_long,
                                mid_forward]

        # cross section @ forward perpencidular
        self.frame_fpp = [bulb_long[0],
                    [block.lwl, 3, 0],
                    [block.lwl, 3, block.draft],
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

    block = Block()
    cp = CtrlPts(block)
    print(cp.cross_frames())
    # cross_frames = CrossSectionFrames(cp.web_frame, cp.transom, cp.frame_fpp)
    # longitudinals = Longitudinals(cp.longitudinal_center)
    # waterlines = Waterlines(cp.waterplane, cp.main_deck)


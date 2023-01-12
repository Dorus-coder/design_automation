from dataclasses import dataclass
import pandas as pd
import numpy as np

df = pd.read_csv("build_vessel\Rules\min_freeboard.csv")
"""
parametric properties of the bulb:
    * bulb length 
    * fullness
    * breadth
    * height
    * nose tip elevation (no influence on Holtrop&Mennen)
    * different shapes?

"""


@dataclass
class Block:
    """Parameters describing the block of the hull shape.
    """
    loa = 205
    lwl = 200
    laft = 25
    lhold = 145
    lfore = lwl - laft - lhold
    boa = 16
    draft = 10
    depth = draft + np.ceil(df.loc[df["length"] == loa].values[0][1] / 1000) # "A" type ship
    bilge_radius = 1
    ctrlpt_offset_forward = 7 # Work with offsets or let the algorithm figure it out?
    # transom
    transom_width = 14 # reduction in breadth at the transom from the main deck
    height_transom = 8.5
    transom_offset = 0
 
    

block = Block()

@dataclass  
class Bulb:
    length = block.loa - block.lwl
    breadth = 3 # 1/2 breadth
    bulb_offset = 0
    HEIGHT = block.draft - bulb_offset



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

bulb = Bulb()

# This would be better placed in a json file 

web_frame = [[block.loa / 2, 0, 0],
             [block.loa / 2, block.boa - block.bilge_radius, 0],
             [block.loa / 2, block.boa, 0],
             [block.loa / 2, block.boa, block.bilge_radius],
             [block.loa / 2, block.boa, block.draft]]

main_deck = [[0, 0, block.depth],
             [0, block.transom_width, block.depth],  
             [block.laft, block.boa, block.depth], 
             [block.laft + block.lhold, block.boa, block.depth], 
             [block.loa, block.boa, block.depth], 
             [block.loa, 0, block.depth]]

# Change 11.48 with function value from finding where the transom intersects with the waterline.
waterplane = [[0, 11.48, block.draft],
              [0, block.boa, block.draft],
              [block.laft, block.boa, block.draft],
              [block.laft + block.lhold, block.boa, block.draft],
              [block.lwl - block.ctrlpt_offset_forward, block.boa, block.draft],
              [block.lwl, 0, block.draft]]

transom = [[0, 0, block.height_transom],
           [0, block.transom_width, block.height_transom - block.transom_offset],
           [0, block.transom_width, block.depth]]



# This is a special list with three sublists containing the first order lines aft, the bspline control points of the bulb and the first order lines fore
bulb_long = [[block.lwl, 0, 0],
             [block.loa, 0, 0],
             [block.loa, 0, block.draft + bulb.bulb_offset], 
             [block.lwl, 0, bulb.HEIGHT]]
mid_forward = [[block.lwl, 0, block.draft],
               [block.loa, 0, block.depth]]
longitudinal_center = [[[0, 0, block.depth],
                        [0, 0, block.height_transom],
                        [block.laft, 0, 0],
                        [block.lwl, 0, 0]],
                         bulb_long,
                         mid_forward]
# cross section @ forward perpencidular
frame_fpp = [bulb_long[0],
             [block.lwl, bulb.breadth, 0],
             [block.lwl, bulb.breadth, bulb.HEIGHT],
             bulb_long[3]]

cross_frames = CrossSectionFrames(web_frame, transom, frame_fpp)
longitudinals = Longitudinals(longitudinal_center)
waterlines = Waterlines(waterplane, main_deck)


"""
This modules contains functions that manipulate stuff

Author: Dorus Boogaard
"""
from copy import deepcopy
import numpy as np



def modify_control_points(ctrlpts : list, xyz : int, value : float):
    """ modifies the x, y or z value of the control points.\n
    Arg:
        ctrlpts (list): control points to be modified.
        xyz (int): x = 0, y = 1, z = 2
        value (float): value to insert in the x, y or z of the control points
    Return:
        new_ctrlpts (list): copy of ctrlpts but modified with value in axis x, y, or z.
    """
    new_ctrlpts = deepcopy(ctrlpts)
    for _, v in enumerate(new_ctrlpts):
        v[xyz] = value
    return new_ctrlpts

def lin_interpolate(arr, x, z_max):
    """The middle control point is the corner of the shape\n
    y(x)  =  y1  +  (x - x1) * (y2 - y1) / (x2 - x1)
    """
    mid1_idx = len(arr[0]) // 2
    mid2_idx = len(arr[1]) // 2
    mid_ctrlpt = arr[0][mid1_idx], arr[1][mid2_idx] 
    x1, x2 = mid_ctrlpt[0][0], mid_ctrlpt[1][0] # All control points are defined for x
    y = mid_ctrlpt[0][1] + (x - x1) * ((mid_ctrlpt[1][1] - mid_ctrlpt[0][1]) / (x2 - x1))
    z = mid_ctrlpt[0][2] + (x - x1) * ((mid_ctrlpt[1][2] - mid_ctrlpt[0][2]) / (x2 -x1))
    
    if len(arr[1]) > len(arr[0]):
        radius_ctrl = arr[1][1], arr[1][3]
    else: radius_ctrl = arr[0][1], arr[0][3]

    y_radius1 = (x - x1) * ((radius_ctrl[0][1]) / (x2 - x1))
    z_radius2 = z_max + (x - x1) * ((radius_ctrl[1][2] - z_max) / (x2 - x1))
    return [[x, 0, z], [x, y_radius1, z], [x, y, z], [x, y, z_radius2], [x, y, z_max]]

def new_cross_fore(waterline : list, wbfrm : list, x : int) -> list:
    wl = np.array(waterline)
    wbfrm = np.array(wbfrm)
    new_frame = deepcopy(wbfrm)
    new_frame[:,0] = wl[:,0][x]
    
    y = new_frame[:,1][::-1]
    diff = wl[:,1][0] - wl[:,1][x]
    for idx, val in enumerate(y):
        if val > wl[:,1][x]:
            y[idx] -= diff
        if y[idx] < 0:
            y[idx] = 0
    new_frame[:,1] = y[::-1]
    return new_frame

from build_vessel.midship import CrossSection
from build_vessel.transom import Transom
from build_vessel.parameters import Block, cross_frames, waterlines
from build_vessel.main import modify_control_points
from pyvista import KochanekSpline, PolyData, Plotter
import numpy as np
from scipy.integrate import simpson, trapezoid, quad

block = Block()

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

def new_cross_fore(waterline : list, wbfrm : list, x : int, samples) -> list:
    """ This function doesn't follow the waterplane exactly. Therefore, use the function located in main.py
    """
    # create numpy array of list of lists
    wl = np.array(waterline)
    wbfrm = np.array(wbfrm)
    
    # interpolate between the input lists and create evenly spaced numpy arrays
    x1, y1 = wl[:,0], wl[:,1]
    f1 = interpolate.interp1d(x1, y1)
    x_even = np.linspace(170, 200, samples)
    y_wl = f1(x_even)
    # x2, y2 = wbfrm[:1], wbfrm[:,2]
    # f2 = interpolate.interp1d(x2, y2)
    # y_wbfrm = f2(x_even)

    # create empty array and set the first column to the x location of the frame
    new_frame = np.empty([samples, 3])
    new_frame[:,0] = wl[:,0][x]
    
    y = wbfrm[:,1][::-1]
    diff = y_wl[0] - y_wl[x]
    for idx, val in enumerate(y):
        # if y val of web frame > y value of the waterline at x.
        if val > y_wl[x]:
            # correct y value of the web frame with the difference between y of webframe and y of the waterline
            y[idx] -= diff
        if y[idx] < 0:
            y[idx] = 0
    new_frame[:,1], new_frame[:,2] = y[::-1], wbfrm[:,2]
    return new_frame

class Area:
    def __init__(self, ul : int, n_frames : int) -> None:
        self.ul = ul
        self._n_evalpts = 100
        self._memory = np.empty([n_frames, self.n_evalpts, 3])
        self.section_area = np.empty([n_frames, 2])

    @property
    def n_evalpts(self):
        return self._n_evalpts

    @n_evalpts.setter
    def n_evalpts(self, number):
        self._n_evalpts = number

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, input : tuple):
        try:
            frame, idx = input
        except:
            raise ValueError("Pass an iterable with row and idx.")
        else:
            self._memory[idx] = np.array(frame)

    def area(self):
        for idx, frame in enumerate(self.memory):
            # z = self.ul - frame[:,2]
            self.section_area[idx] = [frame[0][0], simpson(frame[:,1], frame[:,2], even='last')]

    def volume(self):
        d = self.section_area[1][0] - self.section_area[0][0] 
        print(f"delta : {d}")
        sum_of_arr = 0
        for idx, val in enumerate(self.section_area[:,1]):
            if idx == 0:
                sum_of_arr += val
            elif idx == len(self.section_area[:,1]) - 1:
                sum_of_arr += val
            elif idx % 2 == 0:
                sum_of_arr += val * 2
            elif idx % 2 == 1:
                sum_of_arr += val * 4
        return d  * sum_of_arr / 3


    def volume_scipy(self):
        x = np.linspace(0, self.section_area[:,0][-1] - self.section_area[:,0][0] , self.n_evalpts)
        print(x[-1])
        return simpson(self.section_area[:,1], x)


if __name__ == "__main__":  
    """https://thenavalarch.com/numerical-integration-methods-for-calculating-the-hull-volume/
    """
    import numpy as np
    from build_vessel.waterplane import WaterPlane
    from copy import deepcopy
    import pandas as pd
    from scipy import interpolate


    # waterplane area
    wp = WaterPlane(waterlines.waterplane)
    # wp.delta = 0.001
    wp_points = wp.water_plane_points
    wp_fore = wp.forward

    # FPP with bulb
    bulb = CrossSection(cross_frames.fpp_frame)
    bulb_points = bulb.points()

    # web frame
    wbfrm = CrossSection(cross_frames.web_frame)
    # wbfrm.delta = 0.001
    wbfrm_points = wbfrm.points()

    # midship section fore
    ms_control_points = modify_control_points(wbfrm_points, 0, (block.laft + block.lhold))

    ms_control_points[-1][-1] = block.depth
    ms_1 = CrossSection(ms_control_points)
    ms_2_points = np.array(ms_1.points())

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)

    A = Area(block.depth, len(wp_fore))

    for x in range(len(wp_fore)):
        points = new_cross_fore(wp_fore, ms_2_points, x, len(wp_fore))
        A.memory = (points, x)
        spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
        pl.add_mesh(spline, color="r")
        pl.add_mesh(PolyData(points), color="r", point_size=1, render_points_as_spheres=True)

    A.area()
    print(A.section_area)
    print(f"the volume of the forepeak: {A.volume()}")
    x = np.linspace(0, 30, 100)
    print(f"x outside fun {x[-1]}")
    # print(np.column_stack((x, A.section_area[1])))
    print(f"volume according to scipy {simpson(A.section_area[:,1], x)}")
    print(f"volume according to scipy {A.volume_scipy()}")


    kochanek_spline_1 = KochanekSpline(ms_2_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_3 = KochanekSpline(wp_fore, tension=[t, t, t], continuity=[c, c, c], n_points=1000)

    
    pl.add_mesh(kochanek_spline_1, color='k')
    pl.add_mesh(kochanek_spline_3, color='k')

    pl.add_mesh(PolyData(ms_2_points),
        color="k",
        point_size=1,
        render_points_as_spheres=True,
    )

    pl.add_mesh(PolyData(wp_fore),
    color="k",
    point_size=1,
    render_points_as_spheres=True,
    )

    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()

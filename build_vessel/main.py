from build_vessel.midship import CrossSection
from build_vessel.waterplane import WaterPlane
from build_vessel.main_deck import MainDeck
from build_vessel.longitudinal import Longitudinal
import numpy as np
from pyvista import KochanekSpline, PolyData, Plotter
from dataclasses import dataclass, field
import logging
from build_vessel.transom import Transom
from build_vessel.parameters import block, longitudinals, cross_frames, waterlines
from copy import deepcopy
from scipy.integrate import simpson
from scipy.interpolate import interp1d
logging.basicConfig(filename='vessel_env.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def modify_control_points(ctrlpts : list, xyz : int, value : float):
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
    def memory(self, input : tuple, copy=False):
        if not copy:
            try:
                frame, idx = input
            except:
                raise ValueError("Pass an iterable with row and idx.")
            else:
                self._memory[idx] = np.array(frame)
        else:
            self._memory = input

    def area(self) -> None:
        """calculate the area and set the x location and frame area in self.section_area.
        """
        for idx, frame in enumerate(self.memory):
            # z = self.ul - frame[:,2]
            self.section_area[idx] = [frame[0][0], simpson(frame[:,1], frame[:,2], even='last')]

    def volume(self):
        d = self.section_area[1][0] - self.section_area[0][0] 
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
        x = np.linspace(0, self.section_area[:,0][-1] - self.section_area[:,0][0] , len(self.section_area[:,1]))
        return simpson(self.section_area[:,1], x)

    def statical_moment(self):
        return np.sum(self.section_area[:,0] * self.section_area[:,1])

    def lcb(self):
        return self.statical_moment() / self.volume_scipy()

    def total_area(self):
        return np.sum(self.section_area[:,1])

def prismatic_coefficient(volume : list, area_wbfrm : float):
    return sum(volume) / (area_wbfrm * block.lwl)

@dataclass
class Arguments:
    c_m : float
    c_wp : float 
    transom_area : float
    LTOTAL : float
    LOA : float
    freeboard : float


def main():
    # web frame
    wbfrm = CrossSection(cross_frames.web_frame)
    wbfrm_points = wbfrm.points()

    # midship section aft
    ms_control_points = modify_control_points(cross_frames.web_frame, 0, block.laft)
    ms_1 = CrossSection(ms_control_points)
    ms_1_points = ms_1.points()

    # main deck
    md = MainDeck(block)
    md_points = md.main_deck_points

    # waterplane area
    wp = WaterPlane(waterlines.waterplane)
    wp_points = wp.water_plane_points

    # transom
    transom = Transom(block)
    transom_points = transom.transom_points

    # FPP
    fpp = CrossSection(cross_frames.fpp_frame)
    fpp_points = fpp.points()

    # longitudinal
    long = Longitudinal(longitudinals.center)
    long_points = long.points()


    # print(points_list)
    pl = Plotter()
    c, t, b = (-0.2, 1, 0)

    HEIGHT = block.draft
    # aft
    aft_area = Area(HEIGHT, block.laft)
    for x in range(0, block.laft):
        _ctrpts = lin_interpolate((cross_frames.transom, ms_control_points), x, HEIGHT)
        frame = CrossSection(_ctrpts)
        
        p = frame.points()
        aft_area.memory = (p, x)

        spline = KochanekSpline(p, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
        pl.add_mesh(spline, color="r")
        pl.add_mesh(PolyData(frame.points()), color="r", point_size=1, render_points_as_spheres=True)

    aft_area.area()
    print(f"the area of the web frame {aft_area.section_area[-1]}")
    print(f"the area of the transom frame {transom.area:.2f}")
    print(f"The volume according to my function is {aft_area.volume():.2f}")
    print(f"The volume according to scipy is {aft_area.volume_scipy():.2f}")
    print(f"The statical moment of {aft_area.statical_moment() =}")
    print(f"the {aft_area.total_area() =}")
    print(f"lcb {aft_area.lcb() =:.2f}")
    print()


    # points of the forward waterplane
    wp_fore = wp.forward
    
    # midship section fore
    ms_control_points_2 = modify_control_points(wbfrm_points, 0, (block.laft + block.lhold))

    # ms_control_points_2[-1][-1] = block.depth
    ms_2 = CrossSection(ms_control_points_2)
    ms_2_points = np.array(ms_2.points())
    forward_area = Area(block.depth, len(wp_fore))

    for x in range(len(wp_fore)):
        points = new_cross_fore(wp_fore, ms_2_points, x)
        spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
        pl.add_mesh(spline, color="r")
        pl.add_mesh(PolyData(points), color="r", point_size=1, render_points_as_spheres=True)
        forward_area.memory = (points, x)

    forward_area.area()
    print(f"The volume of the forepeak according to my function is {forward_area.volume()}")
    print(f"The volume of the forepeak according to scipy is {forward_area.volume_scipy()}")
    print(f"the {forward_area.statical_moment() =:.2f}")
    print(f"the {forward_area.lcb() =:.2f}")
    print()

    # Midship
    mid = Area(block.draft, 2)
    mid.memory = ms_1_points, 0
    mid.memory = ms_2_points, 1
    mid.area()
    print(f"the {mid.volume_scipy() =:.2f}")
    print(f"the {mid.statical_moment() =:.2f}")
    print(f"the {mid.lcb() = :.2f}")

    # Total area
    arrays = np.concatenate((aft_area.memory, mid.memory, forward_area.memory), axis=0)
    total_area = Area(block.draft, len(arrays))
    total_area.memory = arrays, True
    total_area.area()
    print(f"{total_area.volume_scipy() =}")
    print(f"{total_area.statical_moment() =}")
    print(f"{total_area.lcb() =}")
    volume_list = [aft_area.volume(), mid.volume_scipy(), forward_area.volume_scipy()]
    c_prism = prismatic_coefficient(volume_list, wbfrm.area)
    print(f"the prismatic coefficient is {c_prism}")

    kochanek_spline_1 = KochanekSpline(wbfrm_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_2 = KochanekSpline(ms_1_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_3 = KochanekSpline(wp_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_4 = KochanekSpline(transom_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_5 = KochanekSpline(md_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_6 = KochanekSpline(long_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    kochanek_spline_7 = KochanekSpline(fpp_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)

    pl.add_mesh(kochanek_spline_1, color='k')
    pl.add_mesh(kochanek_spline_2, color='k')
    pl.add_mesh(kochanek_spline_3, color='k') 
    pl.add_mesh(kochanek_spline_4, color='k')   
    pl.add_mesh(kochanek_spline_5, color='k')   
    pl.add_mesh(kochanek_spline_6,  color='k')   
    pl.add_mesh(kochanek_spline_7,  color='k') 

    pl.add_mesh(PolyData(wbfrm_points),
        color="k",
        point_size=1,
        render_points_as_spheres=True,
    )
    pl.add_mesh(PolyData(fpp_points),
        color="k",
        point_size=1,
        render_points_as_spheres=True,
    )
    pl.add_mesh(PolyData(wp_points),
        color="k",
        point_size=1,
        render_points_as_spheres=True,
    )
    pl.add_mesh(PolyData(transom_points),
    color="k",
    point_size=1,
    render_points_as_spheres=True,
)
    pl.add_mesh(PolyData(md_points),
    color="k",
    point_size=1,
    render_points_as_spheres=True,
)
    pl.add_mesh(PolyData(long_points),
    color="k",
    point_size=1,
    render_points_as_spheres=True,
)

    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()



if __name__ == '__main__':
    main()
  
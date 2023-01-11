import numpy as np
from pyvista import Plotter, Line, CircularArc
from dataclasses import dataclass
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from build_vessel.parameters import control_points, block
from scipy.integrate import simpson, trapezoid, quad
from depricated.BSpline import B_Spline
from copy import deepcopy
from build_vessel.main import modify_control_points

class AftShip(B_Spline):
    def __init__(self) -> None:
        super().__init__()
        self._degree = 2
        self._delta = 0.01
        self._web_frame = []
        self._transom = []
    
    def lin_interpolate(self, arr, x, z_max):
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

    @property
    def area(self) -> float:
        y, z = self.points()[:,1], self.points()[:,2] 
        return trapezoid(y, z)

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta : float):
        self._delta = delta

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree : int):
        self._degree = degree

    @property
    def ctrl_web_frame(self) -> list:
        return self._web_frame

    @ctrl_web_frame.setter
    def ctrl_web_frame(self, ctrlpts  : list) -> None:
        self._web_frame = ctrlpts
    
    @property
    def ctrl_transom(self) -> list:
        return self._transom

    @ctrl_transom.setter
    def ctrl_transom(self, ctrlpts  : list) -> None:
        self._transom = ctrlpts

    def points(self, step : int) -> list:
        """creates 3D points of the aft ship\n
        Arg:
            step (int) : interval of frames along the x-axis.
        Return
            self.frames (list) : list of frames
        """
        for x in range(self.ctrl_transom[0][0], self.ctrl_web_frame[0][0], step):
            self.ctrlpoints = self.lin_interpolate((self.ctrl_transom, self.ctrl_web_frame), x, self.ctrl_transom[-1][-1])
            self.frames = self.b_spline().evalpts
        return self.frames

def midship_coefficient(l, b, r):
    area_circle = np.pi * r ** 2
    small_sqr = r ** 2
    left_over = small_sqr - area_circle / 4
    return (l * b - left_over) / (l * b)

if __name__ == '__main__':
    from pyvista import KochanekSpline, PolyData, Spline, Plotter, Line, CircularArc, MultipleLines

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)
   
    ms_control_points = modify_control_points(control_points.web_frame, 0, block.laft)
    ms_control_points[-1][-1] = block.depth
    aft_ship = AftShip()
    aft_ship.ctrl_web_frame = ms_control_points
    aft_ship.ctrl_transom = control_points.transom
    data_points = aft_ship.points(1)

    print(data_points)
    kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(PolyData(data_points),
        color="k",
        point_size=5,
        render_points_as_spheres=True,
    )
    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()
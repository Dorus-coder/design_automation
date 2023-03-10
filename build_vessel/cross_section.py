"""
This modules creates cross sectional frames

author: Dorus Boogaard
"""
import numpy as np
from geomdl import BSpline
from geomdl import utilities
from scipy.integrate import simpson
from build_vessel.properties import Properties, Info
from build_vessel.utils import lin_interpolate, new_cross_fore
from pyvista import KochanekSpline, PolyData, Plotter

class CrossSection():
    def __init__(self, ctrlpts) -> None:
        self._degree = 2
        self._delta = 0.01
        self._ctrlpts = ctrlpts

    def cross_section_coefficient(self):
        y, z = np.max(np.array(self.points)[:,1]), np.max(np.array(self.points)[:,2])
        return self.area / (y * z)

    @property
    def area(self) -> float:
        points = np.array(self.points)
        y, z = points[:,1], points[:,2] 
        return simpson(y, z)

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
    def ctrlpts(self) -> list:
        return self._ctrlpts

    @ctrlpts.setter
    def ctrlpts(self, ctrlpts  : list) -> None:
        self._ctrlpts = ctrlpts

    def b_spline(self) -> BSpline.Curve:
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = self.ctrlpts
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self.delta 
        return curve

    @property
    def points(self):
        return self.b_spline().evalpts

def midship_coefficient(l, b, r):
    area_circle = np.pi * r ** 2
    small_sqr = r ** 2
    left_over = small_sqr - area_circle / 4
    return (l * b - left_over) / (l * b)


class BuildFrames:
    def __init__(self, waterplane, aftrange: int, height: float) -> None:
        self.wp = waterplane
        self.wp.water_plane_points
        self.aftrange = aftrange
        self.height = height
        self.n_evalpts = 100
        
        self.pl = Plotter()
        self.pl.set_background("royalblue", top="aliceblue")

    def aft(self, laft: int, hold_aft_ctrlpts: list, cross_frames_transom):
        points_array = np.empty([laft, self.n_evalpts, 3])
        for x in np.arange(0, laft):
            _ctrpts = lin_interpolate((cross_frames_transom, hold_aft_ctrlpts), x, self.height)
            frame = CrossSection(_ctrpts)
            
            points = frame.points
            points_array[x] = points
            c, t, b = (-0.2, 1, 0)
            spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
            self.pl.add_mesh(spline, color="r")
            self.pl.add_mesh(PolyData(frame.points), color="r", point_size=1, render_points_as_spheres=True)
        return points_array

    def midship(self, hold_aft_points, hold_fore_points, lmid: int = 2):
        info = Info()
        mid = Properties(self.height, lmid, info)
        mid.memory = np.array(hold_aft_points), 0
        mid.memory = np.array(hold_fore_points), 1
        return mid.memory

    def forward(self, hold_fore_points):
        c, t, b = (-0.2, 1, 0)
        points_array = np.empty([len(self.wp.forward), self.n_evalpts, 3])
        for x in range(len(self.wp.forward)):
            points = new_cross_fore(self.wp.forward, np.array(hold_fore_points), x)
            spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
            self.pl.add_mesh(spline, color="r")
            self.pl.add_mesh(PolyData(points), color="r", point_size=1, render_points_as_spheres=True)
            points_array[x] = points
        return points_array

    def visualize(self):
        self.pl.show_bounds(location='outer', font_size=20, use_2d=True)
        self.pl.show()

    def close_visualisation(self):
        self.pl.close()


if __name__ == '__main__':
    from pyvista import KochanekSpline, PolyData, Spline, Plotter, Line, CircularArc, MultipleLines
    from build_vessel.parameters import cross_frames
    pl = Plotter()
    c, t, b = (-0.2, 1, 0)
    
    ms = CrossSection(cross_frames.fpp_frame)
    print(ms.ctrlpts)
    data_points = ms.points()
    print(f"Area {ms.area}")

    kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(PolyData(data_points),
        color="k",
        point_size=5,
        render_points_as_spheres=True,
    )
    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()
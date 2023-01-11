import numpy as np
from pyvista import Plotter, Line, CircularArc
from dataclasses import dataclass
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from build_vessel.parameters import CrossSectionFrames
from scipy.integrate import simpson, trapezoid, quad


class CrossSection():
    def __init__(self, ctrlpts : CrossSectionFrames) -> None:
        self._degree = 2
        self._delta = 0.01
        self._ctrlpts = ctrlpts

    @property
    def area(self) -> float:
        points = np.array(self.points())
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

    def points(self):
        return self.b_spline().evalpts

def midship_coefficient(l, b, r):
    area_circle = np.pi * r ** 2
    small_sqr = r ** 2
    left_over = small_sqr - area_circle / 4
    return (l * b - left_over) / (l * b)

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
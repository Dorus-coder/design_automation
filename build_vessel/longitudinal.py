import numpy as np
from parameters import Longitudinals
from dataclasses import dataclass
from geomdl import BSpline
from geomdl import utilities

from scipy.integrate import simpson


class Bulb:
    def __init__(self) -> None:
        """This class is depricated.
        """
        self._ctrl_points = []
        self.delta = 0.01

        self._degree = 1

    @property
    def ctrlpts(self) -> list:
        return self._ctrl_points

    @ctrlpts.setter
    def ctrlpts(self, control_points : list) -> None:
        if len(control_points) >= 3:
            self._ctrl_points = control_points
        else: raise ValueError("B_splines require a minimum of three control points")

    @property
    def degree(self) -> int:
        return self._degree

    @degree.setter
    def degree(self, value : int) -> None:
        if 0 < value < len(self.ctrlpts):
            self._degree = value
        else: 
            raise ValueError(f"degree should be > 0 and maximum len(ctrl_points) - 1. max length = {len(self.ctrlpts) - 1}")

    def b_spline(self) -> BSpline.Curve:
        """
        """
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = self.ctrlpts
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self.delta
        return curve

    @property
    def bulb_points(self):
        return self.b_spline().evalpts

class Longitudinal():
    def __init__(self, ctrlpts : Longitudinals) -> None:
        self._points_aft = ctrlpts[0]
        self.bulb_points = ctrlpts[1]
        self._points_fore = ctrlpts[2]
        self.degree = 2 # use setter for this
        self.delta = 0.01

    @property
    def points_aft(self):
        return self._points_aft

    @points_aft.setter
    def points_aft(self, points):
        self._points_aft = points

    @property
    def points_fore(self):
        return self._points_fore

    @points_fore.setter
    def points_fore(self, points):
        self._points_fore = points

    def points(self):
        bulb_points = self.b_spline().evalpts
        return np.vstack((self.points_aft, bulb_points, self.points_fore))

    def b_spline(self) -> BSpline.Curve:
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = self.bulb_points
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self.delta 
        return curve

    def points(self):
        return self.b_spline().evalpts

if __name__ == '__main__':
    from pyvista import KochanekSpline, PolyData, Plotter
    from build_vessel.parameters import longitudinals

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)
   
    long = Longitudinal(longitudinals.center)

    data_points = long.points()
    print(long.bulb_points)
    kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)


    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(PolyData(data_points),
        color="k",
        point_size=10,
        render_points_as_spheres=True,
    )
    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()
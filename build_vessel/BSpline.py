
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from scipy.integrate import simpson
import numpy as np


class B_Spline():
    def __init__(self) -> None:
        self._degree = 2
        self.delta = 0.01
        self._ctrlpts = []

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, delta : float) -> None:
        self._delta = delta

    @property
    def degree(self) -> int:
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

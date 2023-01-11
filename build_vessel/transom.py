import numpy as np
import logging
from build_vessel.parameters import Block
from dataclasses import dataclass
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from scipy.integrate import simpson

class Transom:
    def __init__(self, parameters : Block) -> None:
        self.p = parameters
        self.transom_knots =[[self.p.transom_width, self.p.depth],
                             [self.p.transom_width, self.p.height_transom - self.p.transom_offset],
                             [0, self.p.height_transom]]
        self._delta = 0.01
        self._degree = 2

    @property
    def ctrlpts(self) -> list:
        return self._ctrlpts

    @ctrlpts.setter
    def ctrlpts(self, ctrlpts  : list) -> None:
        self._ctrlpts = ctrlpts
        
    @property
    def area(self) -> float:
        y, z = self.transom_points[:,1], self.transom_points[:,2] 
        return simpson(y[::-1], z[::-1])
    
    def transom(self) -> BSpline.Curve:
        """
        Arg:
            degree (int) : degree < len(ctrlpts)
        """
        curve = BSpline.Curve()
        curve.degree = self._degree
        curve.ctrlpts = self.transom_knots
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self._delta
        return curve       
    
    @property
    def transom_points(self):
        yz = self.transom().evalpts
        x = np.array([0 for _ in yz])
        return np.column_stack((x, yz))

if __name__ == '__main__':
    from pyvista import KochanekSpline, PolyData, Spline, Plotter, Line, CircularArc, MultipleLines

    block = Block()

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)
    
    def closest_value(input_list, input_value):
        arr = np.asarray(input_list)
        i = (np.abs(arr - input_value)).argmin()
        return arr[i]




    transom = Transom(block)
    data_points = transom.transom_points
    print(data_points)
    print(f"The area of the transom is {transom.area}")
    print(f"closetst {closest_value(data_points[:,2], 10)}")
    kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(PolyData(data_points),
        color="k",
        point_size=5,
        render_points_as_spheres=True,
    )
    pl.show_bounds(location='outer', font_size=20, use_2d=True)
    pl.show()
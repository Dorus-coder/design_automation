from parameters import Block
from geomdl import BSpline
from geomdl import utilities
from scipy.integrate import simpson
import numpy as np
block = Block()


class MainDeck:
    def __init__(self, block : Block) -> None:
        self.block = block
        self.main_deck_knots = [[0, 0],
                                [0, self.block.transom_width],  
                                [self.block.laft, self.block.boa], 
                                [self.block.laft + self.block.lhold, self.block.boa], 
                                [self.block.loa, self.block.boa], 
                                # [self.block.loa - self.block.ctrlpt_offset_forward, 0],
                                [self.block.loa, 0]]
        self.delta = 0.01

    def b_spline(self, start : int = None, stop : int = None, degree : int = None) -> BSpline.Curve:
        """
        Arg:
            start (int) : start of the main_deck_points list
            stop (int) : end of the main_deck_points list
        """
        curve = BSpline.Curve()
        if start:
            ctrlpt = self.main_deck_knots[start:]
        elif stop:
            ctrlpt = self.main_deck_knots[:stop]
        if degree:
            curve.degree = degree
        else:
            curve.degree = len(ctrlpt) - 1
        curve.ctrlpts = ctrlpt
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self.delta
        return curve

    @property
    def main_deck_points(self):
        aft = self.b_spline(stop=2).evalpts
        forward = self.b_spline(start=3, degree=2).evalpts
        points = np.vstack((aft, forward))
        z = np.array([self.block.depth for _ in points])
        return np.column_stack((points, z))

if __name__ == '__main__':
	from pyvista import KochanekSpline, PolyData, Plotter

	pl = Plotter()
	c, t, b = (-0.2, 1, 0)

	main_deck = MainDeck(block)
	data_points = main_deck.main_deck_points
	print(data_points[:,0])
	x, y = data_points[:,0], data_points[:,1]
	area = simpson(y, x)
	print(f"The waterplane area is {area}")
	kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)


	pl.add_mesh(kochanek_spline, color='k')
	pl.add_mesh(PolyData(data_points),
		color="k",
		point_size=10,
		render_points_as_spheres=True,
	)
	pl.show_bounds(location='outer', font_size=20, use_2d=True)
	pl.show()
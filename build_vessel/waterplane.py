import numpy as np
import logging
from dataclasses import dataclass
from parameters import Block, waterlines, block
from geomdl import BSpline
from geomdl import utilities
from scipy.integrate import simpson

logging.basicConfig(filename='vessel_env.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log = logging.getLogger(__name__)

class WaterPlane:
	def __init__(self, waterline) -> None:
		self._delta = 0.01
		self.intersect_transom = 11.48
		self.water_plane_ctrl_points = waterline

	@property
	def area(self) -> float:
		x, y = self.water_plane_points[:,0], self.water_plane_points[:,1]
		return simpson(y, x)

	@property
	def c_wp(self) -> float:
		lb = block.lwl * block.boa
		return self.area / lb
		

	@property
	def ctrlpts(self) -> list:
		return self.water_plane_ctrl_points

	@ctrlpts.setter
	def set_ctrlpts(self, ctrlpts  : list) -> None:
		self.water_plane_ctrl_points = ctrlpts

	@property
	def delta(self) -> float:
		return self._delta

	@delta.setter
	def delta(self, delta : float):
		self._delta = delta

	# BSpline methods
	def b_spline(self, start : int = None, stop : int = None, degree : int = None) -> BSpline.Curve:
		"""
		Arg:
			start (int) : start of the main_deck_points list
			stop (int) : end of the main_deck_points list
		"""
		curve = BSpline.Curve()
		if start:
			ctrlpt = self.water_plane_ctrl_points[start:]
		elif stop:
			ctrlpt = self.water_plane_ctrl_points[:stop]
		if degree:
			curve.degree = degree
		else:
			curve.degree = len(ctrlpt) - 1
		curve.ctrlpts = ctrlpt
		curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
		curve.delta = self.delta
		return curve

	@property
	def water_plane_points(self):
		self.aft = self.b_spline(stop=3).evalpts
		self.forward = self.b_spline(start=3, degree=2).evalpts
		return np.vstack((self.aft, self.forward))

if __name__ == '__main__':
	from pyvista import KochanekSpline, PolyData, Spline, Plotter, Line, CircularArc, MultipleLines

	pl = Plotter()
	c, t, b = (-0.2, 1, 0)

	block = Block()
	wp = WaterPlane(block)
	wp.intersect_transom = 11.48 # replaced in main with closest_value() 
	data_points = wp.water_plane_points
	print(data_points[:,1])
	print(f"The waterplane area is {wp.area}")
	print(f"The waterplane area coefficient is {wp.c_wp}")
	kochanek_spline = KochanekSpline(data_points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)


	pl.add_mesh(kochanek_spline, color='k')
	pl.add_mesh(PolyData(data_points),
		color="k",
		point_size=10,
		render_points_as_spheres=True,
	)
	pl.show_bounds(location='outer', font_size=20, use_2d=True)
	pl.show()
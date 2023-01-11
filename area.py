from BSpline import BSpline
from geomdl.visualization import VisMPL


B = BSpline()

B._degree = 2
B._ctrlpts = [[0.0, 0.0], [9, 0], [10.0, 0.0], [10.0, 1.0], [10.0, 10]]
b_spline = B.b_spline()
b_spline.vis = VisMPL.VisCurve3D()

b_spline.b_spline.render()

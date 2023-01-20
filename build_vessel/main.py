from cross_section import CrossSection
from waterplane import WaterPlane
from main_deck import MainDeck
from longitudinal import Longitudinal
import numpy as np
from pyvista import KochanekSpline, PolyData, Plotter
from dataclasses import dataclass
from properties import Properties
from utils import modify_control_points, lin_interpolate, new_cross_fore


@dataclass
class HMInput:
    lpp: float
    B: float
    t_f: float # draft fore
    t_a: float # draft aft
    displ: float
    lcb: float # Longitudinal center of bouyancy in percentage forward of 1/2 lpp
    c_m: float # midship section coefficient 
    c_wp: float # waterplane area coefficient
    a_t: float # transom area
    c_prism: float # prismatic coefficient 
    c_b: float
    ie: float # half angle of entrance 
    velocity: float
    c_stern: int = 0 # stern shape parameter
    # bulb optional
    h_b: float = 0.0001 # centre of bulb area above keel [m]
    a_bt: float = 0.001 # transverse bulb area [m^2]
    h_b: float = 0.0001 # centre of bulb area above keel [m]
    a_bt: float = 0.0001 # transverse bulb area [m^2]

def main():
    from build_vessel.parameters import CtrlPts, Block

    # example vessel
    block = Block()
    cp = CtrlPts(block)

    # web frame
    wbfrm = CrossSection(cp.cross_frames.web_frame)
    wbfrm_points = wbfrm.points()

    # midship section aft
    ms_control_points = modify_control_points(cp.cross_frames.web_frame, 0, block.laft)
    ms_1 = CrossSection(ms_control_points)
    ms_1_points = ms_1.points()

    # main deck
    md = MainDeck(block)
    md_points = md.main_deck_points
 
    # waterplane area
    wp = WaterPlane(cp.waterlines.waterplane)
    wp_points = wp.water_plane_points

    # transom
    transom = CrossSection(cp.cross_frames.transom)
    transom_points = transom.points()
    
    # FPP
    fpp = CrossSection(cp.cross_frames.fpp_frame)
    fpp_points = fpp.points()

    # longitudinal
    long = Longitudinal(cp.longitudinals.center)
    long_points = long.points()

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)


    class BuildFrames:
        def __init__(self, waterplane: WaterPlane, aftrange: int, forwardrange: int, height: float) -> None:
            self.wp = waterplane
            self.aftrange = aftrange
            self.forwardrange = forwardrange

            self.height = height
            self._point_arrays = []

            self.n_evalpts = 100
            self.pl = Plotter()

        def aft(self, laft):
            points_array = np.empty([laft, self.n_evalpts, 3])
            for x in np.arange(0, laft):
                _ctrpts = lin_interpolate((cp.cross_frames.transom, ms_control_points), x, self.height)
                frame = CrossSection(_ctrpts)
                
                points = frame.points()
                points_array[x] = points

                spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
                self.pl.add_mesh(spline, color="r")
                self.pl.add_mesh(PolyData(frame.points()), color="r", point_size=1, render_points_as_spheres=True)
            return points_array

        def midship(self, hold_aft_points: np.ndarray, hold_fore_points: np.ndarray, lmid: int = 2):
            mid = Properties(block.draft, lmid)
            mid.memory = hold_aft_points, 0
            mid.memory = hold_fore_points, 1
            return mid.memory

        def forward(self, hold_fore_points):
            points_array = np.empty([len(wp.forward), self.n_evalpts, 3])
            for x in range(len(wp.forward)):
                points = new_cross_fore(wp.forward, np.array(hold_fore_points), x)
                spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
                pl.add_mesh(spline, color="r")
                pl.add_mesh(PolyData(points), color="r", point_size=1, render_points_as_spheres=True)
                points_array[x] = points
            return points_array

        @property
        def point_arrays(self):
            return self._point_arrays
        
        @point_arrays.setter
        def point_arrays(self, hold_aft_points, hold_fore_points):
            self._point_arrays = np.concatenate((self.aft(), self.midship(hold_aft_points, hold_fore_points), self.forward(hold_fore_points)), axis=0)

    # Total area
    arrays = np.concatenate((aft_area.memory, mid.memory, forward_area.memory), axis=0)
    total_area = Properties(block.draft, len(arrays))
    total_area.memory = arrays, True
    total_area.area()
    print(f"{total_area.volume_scipy() = :.2f}")
    print(f"{total_area.statical_moment() = :.2f}")
    print(f"{total_area.lcb() =}")
    print(f"immersed {total_area.transom_area = }")
    c_prism = total_area.prismatic_coefficient(wbfrm.area, block.lwl)
    print(f"the prismatic coefficient is {c_prism = :.2f}")
    print(f"{total_area.ie(block.boa, block.lfore) = }")
    print(f"{wbfrm.cross_section_coefficient() = }")
    
    # Holtrop and Mennen input
    hm_input = HMInput(lpp= block.lwl,
                       B= block.boa * 2,
                       t_f= block.draft,
                       t_a= block.draft,
                       displ= total_area.volume_scipy() * 2 * 1.025,
                       lcb=total_area.lcb(),
                       c_m=wbfrm.cross_section_coefficient(),
                       c_wp=wp.c_wp(block.lwl, block.boa),
                       c_b=total_area.block_coefficient(block.lwl, block.boa, block.draft),
                       a_t=total_area.transom_area,
                       c_prism= total_area.prismatic_coefficient(wbfrm.area, block.lwl),
                       ie=total_area.ie(block.boa, block.lfore), 
                       velocity=18                
                      )
    from HoltropMennen import HoltropMennen
    resistance = HoltropMennen(hm_input)
    print(f"1. {resistance.total_resistance() = :.2f}")
    print(f"2. {resistance.bulb_res() = :.2f}")

    # Visualization
    def visualize(plot_points: list, plotter: Plotter):
        # global pl
        for plot_point in plot_points:
                spline = KochanekSpline(plot_point, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
                plotter.add_mesh(spline, color='k')
                plotter.add_mesh(PolyData(plot_point),
                color="k",
                point_size=1,
                render_points_as_spheres=True,
                                )
                pl.show_bounds(location='outer', font_size=20, use_2d=True)
                pl.show()

    # visualize([wbfrm_points, ms_1_points, wp_points, transom_points, md_points, long_points, fpp_points], pl)                         
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
  
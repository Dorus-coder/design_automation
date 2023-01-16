from cross_section import CrossSection
from waterplane import WaterPlane
from main_deck import MainDeck
from longitudinal import Longitudinal
import numpy as np
from pyvista import KochanekSpline, PolyData, Plotter
from dataclasses import dataclass
from parameters import block, longitudinals, cross_frames, waterlines
from properties import Properties
from helpers import modify_control_points, lin_interpolate, new_cross_fore


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
    # web frame
    wbfrm = CrossSection(cross_frames.web_frame)
    wbfrm_points = wbfrm.points()

    # midship section aft
    ms_control_points = modify_control_points(cross_frames.web_frame, 0, block.laft)
    ms_1 = CrossSection(ms_control_points)
    ms_1_points = ms_1.points()

    # main deck
    md = MainDeck(block)
    md_points = md.main_deck_points

    # waterplane area
    wp = WaterPlane(waterlines.waterplane)
    wp_points = wp.water_plane_points

    # transom
    transom = CrossSection(cross_frames.transom)
    transom_points = transom.points()
    
    # FPP
    fpp = CrossSection(cross_frames.fpp_frame)
    fpp_points = fpp.points()

    # longitudinal
    long = Longitudinal(longitudinals.center)
    long_points = long.points()

    pl = Plotter()
    c, t, b = (-0.2, 1, 0)

    HEIGHT = block.draft
    
    # aft
    aft_area = Properties(HEIGHT, block.laft)
    for x in range(0, block.laft):
        _ctrpts = lin_interpolate((cross_frames.transom, ms_control_points), x, HEIGHT)
        frame = CrossSection(_ctrpts)
        
        p = frame.points()
        aft_area.memory = (p, x)

        spline = KochanekSpline(p, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
        pl.add_mesh(spline, color="r")
        pl.add_mesh(PolyData(frame.points()), color="r", point_size=1, render_points_as_spheres=True)

    aft_area.area()
    print(f"the area of the web frame {aft_area.section_area[-1]}")
    print(f"the area of the transom frame {transom.area:.2f}")
    print(f"The volume according to my function is {aft_area.volume():.2f}")
    print(f"The volume according to scipy is {aft_area.volume_scipy():.2f}")
    print(f"The statical moment of {aft_area.statical_moment() =}")
    print(f"the {aft_area.total_area() =}")
    print(f"lcb {aft_area.lcb() =:.2f}")
    print()

    # points of the forward waterplane
    wp_fore = wp.forward
    
    # midship section fore
    ms_control_points_2 = modify_control_points(wbfrm_points, 0, (block.laft + block.lhold))

    # ms_control_points_2[-1][-1] = block.depth
    ms_2 = CrossSection(ms_control_points_2)
    ms_2_points = np.array(ms_2.points())
    forward_area = Properties(block.depth, len(wp_fore))

    for x in range(len(wp_fore)):
        points = new_cross_fore(wp_fore, ms_2_points, x)
        spline = KochanekSpline(points, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
        pl.add_mesh(spline, color="r")
        pl.add_mesh(PolyData(points), color="r", point_size=1, render_points_as_spheres=True)
        forward_area.memory = (points, x)

    forward_area.area()
    print(f"The volume of the forepeak according to my function is {forward_area.volume()}")
    print(f"The volume of the forepeak according to scipy is {forward_area.volume_scipy()}")
    print(f"the {forward_area.statical_moment() =:.2f}")
    print(f"the {forward_area.lcb() =:.2f}")
    print()

    # Midship
    mid = Properties(block.draft, 2)
    mid.memory = ms_1_points, 0
    mid.memory = ms_2_points, 1
    mid.area()
    print(f"the {mid.volume_scipy() =:.2f}")
    print(f"the {mid.statical_moment() =:.2f}")
    print(f"the {mid.lcb() = :.2f}")

    # Total area
    arrays = np.concatenate((aft_area.memory, mid.memory, forward_area.memory), axis=0)
    total_area = Properties(block.draft, len(arrays))
    total_area.memory = arrays, True
    total_area.area()
    print(f"{total_area.volume_scipy() = :.2f}")
    print(f"{total_area.statical_moment() = :.2f}")
    print(f"{total_area.lcb() =}")
    print(f"immersed {total_area.transom_area = }")
    c_prism = total_area.prismatic_coefficient(wbfrm.area)
    print(f"the prismatic coefficient is {c_prism = :.2f}")
    print(f"{total_area.ie() = }")
    print(f"{wbfrm.cross_section_coefficient() = }")
    
    # Holtrop and Mennen input
    hm_input = HMInput(lpp= block.lwl,
                       B= block.boa * 2,
                       t_f= block.draft,
                       t_a= block.draft,
                       displ= total_area.volume_scipy() * 2 * 1.025,
                       lcb=total_area.lcb(),
                       c_m=wbfrm.cross_section_coefficient(),
                       c_wp=wp.c_wp,
                       c_b=total_area.block_coefficient(),
                       a_t=total_area.transom_area,
                       c_prism= total_area.prismatic_coefficient(wbfrm.area),
                       ie=total_area.ie(), 
                       velocity=18                
                      )
    from HoltropMennen import HoltropMennen
    resistance = HoltropMennen(hm_input)
    print(f"1. {resistance.total_resistance() = :.2f}")
    print(f"2. {resistance.bulb_res() = :.2f}")

    # Visualization
    def visualize(plot_points: list, plotter: Plotter):
        for plot_point in plot_points:
                KochanekSpline(plot_point, tension=[t, t, t], continuity=[c, c, c], n_points=1000)
                plotter.add_mesh(plot_point, color='k')
                plotter.add_mesh(PolyData(wbfrm_points),
                color="k",
                point_size=1,
                render_points_as_spheres=True,
                                )

                                
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
  
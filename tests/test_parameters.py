from dataclasses import dataclass
from build_vessel.parameters import Block

@dataclass
class ship_charasteristics:
    loa = 205.0
    lpp = 200
    B  = 32
    t_f = 10
    t_a = 10
    displ = 37_500
    lcb = -2.02
    a_bt = 20 # transverse bulb area [m^2]
    h_b = 4 # centre of bulb area above keel [m]
    c_m = 0.98 # midship section coefficient 
    c_wp = 0.75 # waterplane area coefficient
    a_t = 16 # transom area 
    s_app = 50 # wetted area appendages
    c_stern = 10 # stern shape parameter
    d_prop = 8 # propellor diameter [m]
    z = 4 # number of blades 
    clear_prop = 0.2 # clearance propellor with keel line
    velocity = 25 # [knots]

coef = Block()
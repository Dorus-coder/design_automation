import numpy as np
from dataclasses import dataclass
import random

@dataclass
class VarParameters:
    # This class is the enviroment in which the agent can act. During this stage I will constrains the vriables and initialize them randomly
    l_hold : float
    t_delta_fore : float # -/+ from mean draft [% of mean draft]
    t_delta_aft : float # -/+ from mean draft [% of mean draft]
    lcb : float 
    c_b : float
    c_w : float
    ie : float # angle of the waterline with the bow
    c_stern : int # stern shape parameter

@dataclass(frozen=True)
class ConParameters:
    # Here are parameters stored that the are immutable
    DISPL : int = 46916 # [ton]
    DEADWEIGHT : int = 38981 # [ton]
    BALE : int = 47849 #m^3
    FORE : int = 30 # [m]
    AFT : int = 40 # [m]
    SIDE : float = 2.0 # [m]
    GAMMA = 0.1883
    G  = 9.81
    RHO = 1.025 # [kg/m^3]
    VEL : float = 12.86 # m/s

class MainDimensions:
    def __init__(self, constants : ConParameters, variables : VarParameters) -> None:
        # this way of defining variables, which has quit long names?
        self.var = variables
        self.con = constants
        
        # or this way of declaring variables which is a quit long list?
        self.DISPL = constants.DISPL
        self.BALE = constants.BALE
        self.LFORE = constants.FORE
        self.LAFT = constants.AFT
        self.SIDE = constants.SIDE
        self.cb = variables.c_b
        self.l_hold = variables.l_hold

        # 
        self.loa, self.boa, self.height = self._maindim()

    def _maindim(self):
        b_min = self.l_hold / 9
        b_max = self.l_hold / 4
        b_hold = random.uniform(b_min, b_max)
        height = self.BALE / (self.l_hold * b_hold)
        loa = self.l_hold + self.LFORE + self.LAFT
        boa = b_hold + self.SIDE
        return loa, boa, height

    @property
    def draught_mean(self):
        return self.DISPL / (self.loa * self.boa * self.cb)

    @property
    def draught_fore(self):
        return self.draught_mean + self.var.t_delta_fore * self.draught_mean

    @property
    def draught_aft(self):
        return self.draught_mean + self.var.t_delta_aft * self.draught_mean

@dataclass
class ResistanceParameters:
    form_eq : float # form factor equivalent
    form_app : float # form factor appendages

@dataclass
class Ship_Charasteristics:
    loa : MainDimensions
    lpp : float
    B  : MainDimensions
    t_f : MainDimensions
    t_a : MainDimensions
    displ : ConParameters
    lcb :  VarParameters
    a_bt : float # transverse bulb area [m^2]
    h_b : float # centre of bulb area above keel [m]
    c_m : float # midship section coefficient 
    c_wp : float # waterplane area coefficient
    a_t : float # transom area 
    s_app : float
    c_stern : VarParameters
    d_prop : float # propellor diameter [m]
    z : int # number of blades 
    clear_prop : float = 0.2 # clearance propellor with keel line
    velocity : float = 25.0 # [knots]


def random_var(*args) -> VarParameters:
    var_data = {}
    for input in args:
        if input == 'l_hold':
           l_hold = random.uniform(0, 500)
        elif input == 't_delta_fore':
            trim = random.uniform(0, 0.25)
            t_delta_fore = trim / 2
            t_delta_aft = trim / 2
        elif input == 'lcb':
            lcb = random.uniform(-5, 5)
        elif input == 'c_b':
            c_b = random.uniform(0.6, 0.85)
        elif input == 'c_w':
            c_w = random.uniform(0.6, 0.85)
        elif input == 'ie':
            ie = random.uniform(0, 90)
        elif input == 'c_stern':
            shape = [-10, 0, 10]
            c_stern = shape[random.randint(0, len(shape) - 1)]
    return l_hold, t_delta_fore , t_delta_aft, lcb, c_b, c_w, ie, c_stern

def print_dataclass(dataclass):
    name = type(dataclass).__name__
    print(f'{name} has the following parameters and arguments')
    for key, value in dataclass.__dict__.items():
        print(f'{key} : {value}')

CON = ConParameters()
a, b, c, d, e, f, g, h = random_var('l_hold', 't_delta_fore', 'lcb', 'c_b', 'c_w', 'ie', 'c_stern')
var = VarParameters(a, b, c, d, e, f, g, h)
main_dims = MainDimensions(CON, var)

ship_char = Ship_Charasteristics(main_dims.loa, main_dims.loa, main_dims.boa, main_dims.draught_fore, main_dims.draught_aft, 
                                CON.DISPL, var.lcb, 20, 4, 0.98, 0.75, 16, 50, var.c_stern, 8.0, 4)                                
print_dataclass(ship_char)
class Coefficients:
    def __init__(self, constants : ConParameters, var : VarParameters) -> None:
        self.CON = constants
        self.var = var
    
    @property
    def reynolds(self):
        """ 
        Arg:
            dviscosity (constant) : dynamic viscosity 
        """
        return self.con.VEL * self.var.L / ConParameters.GAMMA

    def froude(self):
        return self.v / np.sqrt(ConParameters.G * self.l)
    
    def friction_coefficient(self):
        return 0.075 / (np.log10(self.reynolds) - 2) ** 2

@dataclass
class InputParameters:
    dimensions : MainDimensions 
    form : ResistanceParameters 
    coefficients : Coefficients



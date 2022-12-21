import numpy as np
from dataclasses import dataclass
import random

DISPL = 46916
DEADWEIGHT = 38981 
BALE = 47849 #m^3
FORE = 30
AFT = 40
SIDE = 2

class MainDimension:
    def __init__(self, deplacement, bale, length_fore, length_aft, side, cb) -> None:
        self.depl = deplacement
        self.bale = bale
        self.l_fore = length_fore
        self.l_aft = length_aft
        self.side = side
        self.cb = cb

    def maindim(self):
        l_hold = random.randint(0, 500)
        b_min = l_hold / 9
        b_max = l_hold / 4
        b_hold = random.uniform(b_min, b_max)
        height = self.bale / (l_hold * b_hold)
        loa = l_hold + self.l_fore + self.l_aft
        boa = b_hold + self.side
        return loa, boa, height

    def draught(self, cb):
        loa, boa, _ = self.maindim()
        return self.depl / (loa * boa * cb)



@dataclass
class Constants:
    GAMMA = 0.1883
    G  = 9.81
    RHO = 1.025

@dataclass
class Parameters:
    l : float # length
    b : float # breadth
    t : float # draft mean
    tf : float # draft forward
    ta : float # draft aft
    v : float # velocity
    sapp : float # wetted surface appendages
    am : float # midship section
    wpa : float # waterplane area
    lcb : float # longitudinal center of bouyancy
    at : float # Immersed part of the transverse area at zero speed
    ie : float # angle of the waterline with the bow
    hb : float # center of the transverse area above the keel line
    cstern : int # stern shape parameter

@dataclass
class Coefficients:
    cb : float     


@dataclass
class ResistanceParameters:
    form_eq : float # form factor equivalent
    form_app : float # form factor appendages

class DimensionlessParameters:
    def __init__(self, length, velocity) -> None:
        self.l = length
        self.v = velocity   
       
    def reynolds(self):
        """ 
        Arg:
            dviscosity (constant) : dynamic viscosity 
        """
        return self.v * self.l / Constants.GAMMA

    def froude(self):
        return self.v / np.sqrt(Constants.G * self.l)
    
class some:
    def __init__(self) -> None:
        self.p = Parameters
        self.dp = DimensionlessParameters
        self.CONSTANTS = Constants
        self.c = Coefficients



    def ca(self):
        # gives atm a negative outcome. Search for other formulas that account for the hull roughness.
        "calculates the correlation allowance coefficient Ca."
        return 0.006 * (self.p.l + 100) ** -16 - 0.00205 + 0.003 * np.sqrt(self.p.l / 7.5) * self.c **4 * c2() * \
    (0.04 - c4(TF, L))
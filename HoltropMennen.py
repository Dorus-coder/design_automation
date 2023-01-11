import numpy as np
import math

class ShipCharasteristics:
    loa = 205.0
    lpp = 200
    B  = 32
    t_f = 10
    t_a = 10
    displ = 37_500
    lcb = -0.75#-2.02
    a_bt = 20 # transverse bulb area [m^2]
    h_b = 4 # centre of bulb area above keel [m]
    c_m = 0.98 # midship section coefficient 
    c_wp = 0.75 # waterplane area coefficient
    a_t = 16 # transom area 
    s_app = 50 # wetted area appendages
    c_stern = 10 # stern shape parameter
    c_prism = 0.5833  #0.5477 # prismatic coefficient 
    c_b = 0.612
    ie = 12.08 # half angle of entrance 
    # propellor 
    d_prop = 8 # propellor diameter [m]
    z = 4 # number of blades 
    clear_prop = 0.2 # clearance propellor with keel line
    velocity = 25 # [knots]


class HoltropMennen:
    def __init__(self): 
        self.ship = ShipCharasteristics()
        self.mean_draft = (self.ship.t_a + self.ship.t_f) / 2
        self._lwl = self.ship.lpp 
        self.velocity = self.kn_to_ms(self.ship.velocity)
        # constants
        self.RHO = 1.025
        self.G = 9.81
        self.GAMMA = 0.1883

    def friction_res(self):
        """frictional resistance according to the ITTC-1957 formula
        """
        return self.cf * 0.5 * self.RHO * self.velocity ** 2 * self.wetted_area()

    def wave_res(self):
        """Wave Resistance
        m1 -2.1274
        m2 -0.17087
        test case = 557.11 kN
        """
        # to degree test case differ +/- 45 N with degrees or radians
        RAD = 180 / np.pi
        D = -0.9
        m1 = 0.0_140_407 * self.lwl / self.mean_draft - 1.75_254 * self.ship.displ ** (1 / 3) / self.lwl + (-4.79_323 * self.ship.B / self.lwl) - self.c16
        m2 = self.c15 * self.ship.c_prism ** 2 * np.exp(-0.1 * self.fn ** -2)
        return self.c1 * self.c2 * self.c5 * self.ship.displ * self.RHO * self.G * np.exp(m1 * self.fn ** D + m2 * np.cos(self.lambd * self.fn ** -2)) 

    def bulb_res(self):
        """bulb resistance\n
        test case = 0.049 kN
        """ 
        return 0.11 * np.exp(-3 * self.p_b ** -2) * self.fn_i ** 3 * self.ship.a_bt ** 1.5 * self.RHO * self.G / (1 + self.fn_i ** 2)

    def corr_res(self):
        """model-ship correlation resistance
        """
        return 0.5 * self.RHO * self.velocity ** 2 * self.wetted_area() 

    def rtr(self):
        """additional resistance from due to the immersed transom.
        test case 0
        """
        return 0.5 * self.RHO * self.velocity ** 2 * self.ship.a_t * self.c6

    def ra(self):
        """Describes primarily the effect of the hull roughness and still air resistance.
        test case 221.98
        """
        # self.c_a() = 0.000_352 this cause major problems
        return self.c_a() * 0.5 * self.RHO * self.velocity ** 2 * self.wetted_area()
   

    @property
    def cf(self):
        """the frictional resistance coefficient\n
        test case = 0.001390
        """
        return 0.075 / ((np.log(self.rn) - 2) ** 2)
    
    @property
    def rn(self):
        """Reynolds number
        """
        Rn = self.velocity * self.lwl / self.GAMMA
        return Rn
        
    @property
    def fn(self):
        """Froude number
        """
        return self.velocity / np.sqrt(self.G * self.lwl)

    def kn_to_ms(self, kn):
        return kn * 1852 / 3600

    @property
    def lwl(self):
        return self._lwl

    def form_factor(self):
        """form factor 1 + k1\n
        L represents the waterline length.\n
        test case Lr = 81.385, 1+k1 = 1.156
        Return:
            (float) : 1+k1 
        """
        lr = (1 - self.ship.c_prism + 0.06 * self.ship.c_prism * self.ship.lcb / (4 * self.ship.c_prism - 1)) * self.ship.lpp
        # lr = 81.385
        return self.c13 * (0.93 + self.c12 * (self.ship.B / lr) ** 0.92497 * (0.95 - self.ship.c_prism) ** -0.521448 * (1 - self.ship.c_prism + 0.0225 * self.ship.lcb) ** 0.6906)

    def wetted_area(self):
        """This function apprioximates the wetted surface\n
        In the future calculate the wetted area based on the section frames
        """
        return self.lwl * (2 * self.mean_draft + self.ship.B) * np.sqrt(self.ship.c_m) * (0.453 + 0.4425 * self.ship.c_b - 0.2862 * self.ship.c_m - 0.003467 * self.ship.B / self.mean_draft + 0.3696 * self.ship.c_wp) + 2.38 * self.ship.a_bt / self.ship.c_b

    def rapp(self):
        """Resistance from appendages, which the vessel doesn't have at the moment.
        """
        return

    @property
    def lambd(self):
        """lambda
        test = 0.6513
        """
        ratio = self.lwl / self.ship.B
        if ratio < 12:
            return 1.446 * self.ship.c_prism - 0.03 * ratio
        else:
            return 1.446 * self.ship.c_prism - 0.36

    @property
    def c1(self):
        #1.398
        return 2_223_105 * self.c7 ** 3.78_613 * (self.mean_draft / self.ship.B) ** 1.07_961 * (90 - self.ship.ie) ** -1.37_565

    @property
    def c2(self):
        """reduction of wave resistance due to the action of bulbous bow.
        0.7595 checks with the right c3 diff =/- 0.02
        """
        return math.exp(-1.89 * np.sqrt(self.c3))
    
    @property
    def c3(self):
        """influence of the bulbous
        0.02119
        """
        return 0.56 * self.ship.a_bt ** 1.5 / (self.ship.B * self.mean_draft * (0.31 * np.sqrt(self.ship.a_bt) * self.ship.t_f - self.ship.h_b))

    @property
    def c4(self):
        """test case = 0.04
        """
        ratio = self.ship.t_f / self.lwl
        if ratio <= 0.04:
            return ratio
        else:
            return 0.04

    @property
    def c5(self):
        """influence of the transom stern on the wave resistance
        0.9592
        """
        return 1 - 0.8 * self.ship.a_t / (self.ship.B * self.mean_draft * self.ship.c_m)

    @property
    def c6(self):
        """Related to the Froude number based on the transom immersion\n
        """
        if self.fn_t < 5:
           return 0.2 * (1 - 0.2 * self.fn_t)
        else:
            return 0
    
    @property
    def c7(self):
        # 0.1561 in test case, has significant impact on c1.
        ratio = self.ship.B / self.lwl
        if ratio < 0.11:
            return 0.229_577 * ratio  ** 0.33_333
        elif ratio < 0.25:
            return ratio
        else:
            return 0.5 - 0.0625 * ratio

    @property
    def c12(self):
        """calculates the coefficient c12.
        0.5102 for the testcase
        """
        ratio = self.mean_draft / self.lwl
        if ratio < 0.02:
            return 0.479948
        elif ratio < 0.05:
            return 48.2 * (ratio - 0.02) ** 2.078 + 0.479948
        else:
            return ratio ** 0.2228446
       
    @property
    def c13(self):
        """
        C_stern (int) :\n\t\tV-shaped section -10\n
        \tNormal shaped section 0\n
        \tU-shaped section with hogner stern 10 \n
        """
        return 1 + 0.003 * self.ship.c_stern

    @property
    def c15(self):
        """1.69_385
        """
        ratio = self.lwl ** 3 / self.ship.displ
        if ratio < 512:
            return -1.69_385 # Holtrop and mennen is not clear about wheter this value is negative or positive
        elif ratio < 1727:
            return -1.69_385 + (self.lwl / self.ship.displ ** (1/3) -8) / 2.36
        else:
            return 0
            
    @property
    def c16(self):
        if self.ship.c_prism < 0.8:
            return 8.07_981 * self.ship.c_prism - 13.8_673 * self.ship.c_prism ** 2 + 6.984_388 * self.ship.c_prism ** 3
        else:
            return 1.73_014 - 0.7_067 * self.ship.c_prism

    @property
    def p_b(self):
        """measure of emergence of the bow\n
        test case 0.6261
        """
        return 0.56 * np.sqrt(self.ship.a_bt) / (self.ship.t_f - 1.5 * self.ship.h_b)

    @property
    def fn_i(self):
        """Froude number based on the immersion\n
        test case 1.5084\n
        """
        return self.velocity / np.sqrt(self.G * (self.ship.t_f - self.ship.h_b - 0.25 * np.sqrt(self.ship.a_bt)) + 0.15 * self.velocity ** 2)

    @property
    def fn_t(self):
        """Froude number based on the transom area
        testcase 5.433
        """
        return self.velocity / np.sqrt(2 * self.G * self.ship.a_t / (self.ship.B + self.ship.B * self.ship.c_wp))

    def c_a(self, increase=0.0):
        """correlation allowance coefficient\n
        test case 0.000_352 which is *10 smaller and positive maybe some parenthesis are wrong\n
        Arg:
            increase (float) : could be increased to take the effect of a larger hull roughness in account \n
        Return:
            (float) : c_A from 19th ITTC
        """
        # 0.006 * (self.lwl + 100) ** -16 - 0.00_205 + 0.003 * np.sqrt(self.lwl / 7.5) * self.ship.c_b ** 4 * self.c2 * (0.04 - self.c4) + increase
        # above equation causes major problems 
        return (5.68 - 0.6 * np.log10(self.rn)) * 10 ** -4

    

hm = HoltropMennen()
print(f"{hm.c_a() = }")
print(f"{hm.ra() = }")
print(f"{hm.cf = }")
print(f"{hm.rn = }")
print(f"{np.log10(hm.rn) = }")
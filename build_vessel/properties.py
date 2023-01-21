"""
This module contains methods that calculate the properties of the ship.

author: Dorus Boogaard
"""
import numpy as np
from scipy.integrate import simpson

class Properties:
    def __init__(self, ul : int, n_frames : int, info) -> None:
        self.info = info
        self.ul = ul
        self._n_evalpts = 100
        self._memory = np.empty([n_frames, self.n_evalpts, 3])
        self.section_area = np.empty([n_frames, 2])

    @property
    def n_evalpts(self) -> int:
        return self._n_evalpts

    @n_evalpts.setter
    def n_evalpts(self, number) -> None:
        self._n_evalpts = number

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, input : tuple, copy=False):
        if not copy:
            try:
                frame, idx = input
            except:
                raise ValueError("Pass an iterable with row and idx.")
            else:
                self._memory[idx] = np.array(frame)
        else:
            self._memory = input

    def area(self) -> None:
        """calculate the area and set the x location and frame area in self.section_area.
        """
        for idx, frame in enumerate(self.memory):
            # z = self.ul - frame[:,2]
            self.section_area[idx] = [frame[0][0], simpson(frame[:,1], frame[:,2], even='last')]
    
    @property
    def transom_area(self):
        """immersed transom area
        """
        area = self.section_area[0][1]
        self.info.transom_area = area
        return area

    def volume(self) -> float:
        """This function is soon to be depricated.
        """
        d = self.section_area[1][0] - self.section_area[0][0] 
        sum_of_arr = 0

        for idx, val in enumerate(self.section_area[:,1]):
            if idx == 0:
                sum_of_arr += val
            elif idx == len(self.section_area[:,1]) - 1:
                sum_of_arr += val
            elif idx % 2 == 0:
                sum_of_arr += val * 2
            elif idx % 2 == 1:
                sum_of_arr += val * 4
        return d  * sum_of_arr / 3

    def volume_scipy(self) -> float:
        x = np.linspace(0, self.section_area[:,0][-1] - self.section_area[:,0][0] , len(self.section_area[:,1]))
        volume = simpson(self.section_area[:,1], x)
        self.info.volume = volume
        return volume

    def statical_moment(self) -> float:
        statical_moment = np.sum(self.section_area[:,0] * self.section_area[:,1])
        self.info.statical_moment = statical_moment
        return statical_moment

    def lcb(self) -> float:
        lcb = self.statical_moment() / self.volume_scipy()
        self.info.lcb = lcb
        return lcb

    def total_area(self) -> float:
        return np.sum(self.section_area[:,1])

    def prismatic_coefficient(self, area_wbfrm : float, lwl: float) -> float:
        c_p = self.volume_scipy() / (area_wbfrm * lwl)
        self.info.prismatic_coefficient = c_p
        return c_p

    def block_coefficient(self, lwl, boa, draft) -> float:
        """Block coefficient defined at the draft.
        """
        c_b = self.volume_scipy() / (lwl * boa * draft)
        self.info.block_coefficient = c_b
        return c_b

    def ie(self, boa, lfore):
        """half angle of entrance in degree
        """
        RAD = 180 / np.pi
        ie = np.tanh(boa / lfore) * RAD
        self.info.ie = ie
        return ie

class Info:
    def __init__(self) -> None:
        self.laft = None
        self.lhold = None
        self.lfore = None
        self.lwl = None
        self.half_boa = None
        self.draft = None
        self.transom_area = None
        self.volume = None
        self.statical_moment = None
        self.lcb = None
        self.prismatic_coefficient = None
        self.block_coefficient = None
        self.ie = None

    def __str__(self) -> str:
        return f"{self.transom_area = :.2f} \n {self.volume = :.2f} \n {self.statical_moment = :.2f} \n {self.lcb = :.2f} \n {self.prismatic_coefficient = :.2f} \n {self.block_coefficient = :.2f} \n {self.ie = :.2f}"

    def __repr__(self) -> str:
        return f'Info(transom_area={self.transom_area}, volume={self.volume}, statical_moment={self.statical_moment}, lcb={self.lcb}, prismatic_coefficient={self.prismatic_coefficient}, block_coefficient={self.block_coefficient}, ie={self.ie})'
    
    def __lwl(self) -> float:
        if self.laft != None and self.lhold != None and self.lfore != None:
            return sum((self.laft, self.lhold, self.lfore))
        else:
            return None  

    def print_info(self) -> dict:
        for key, value in self.__dict__.items():
            print(f"{key} = {value}")
    

if __name__ == "__main__":
    info = Info()
    print(info)
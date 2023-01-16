"""
This module contains methods that calculate the properties of the ship.

author: Dorus Boogaard
"""

import numpy as np
from build_vessel.parameters import block
from scipy.integrate import simpson

class Properties:
    def __init__(self, ul : int, n_frames : int) -> None:
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
        return self.section_area[0][1]

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
        return simpson(self.section_area[:,1], x)

    def statical_moment(self) -> float:
        return np.sum(self.section_area[:,0] * self.section_area[:,1])

    def lcb(self) -> float:
        return self.statical_moment() / self.volume_scipy()

    def total_area(self) -> float:
        return np.sum(self.section_area[:,1])

    def prismatic_coefficient(self, area_wbfrm : float) -> float:
        return self.volume_scipy() / (area_wbfrm * block.lwl)

    def block_coefficient(self) -> float:
        """Block coefficient defined at the draft.
        """
        return self.volume_scipy() / (block.lwl * block.boa * block.draft)

    def ie(self):
        """half angle of entrance in degree
        """
        RAD = 180 / np.pi
        return np.tanh(block.boa / block.lfore) * RAD

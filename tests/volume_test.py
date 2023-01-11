import unittest
import numpy as np
# from build_vessel.interpolate_frames import volume

def volume(arr):
    d = (arr[1][0] - arr[0][0]) 
    arr = arr
    sum_of_arr = 0

    for idx, val in enumerate(arr[:,1]):
        if idx == 0:
            sum_of_arr += val
        elif idx == len(arr[:,1]) - 1:
            sum_of_arr += val
        elif idx % 2 == 0:
            sum_of_arr += val * 2
        elif idx % 2 == 1:
            sum_of_arr += val * 4
    return d  * sum_of_arr / 3


class TestVolume(unittest.TestCase):
    def test_volume(self):
        # Test 1: Test the volume of a simple object with a constant cross-sectional area
        arr = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])
        result = volume(arr)
        self.assertEqual(result, 4.0)

        # Test 2: Test the volume of a simple object with a linearly increasing cross-sectional area
        arr = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        result = volume(arr)
        self.assertEqual(result, 8.0)

        # # Test 3: Test the volume of a simple object with a linearly decreasing cross-sectional area
        # arr = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]])
        # result = volume(arr)
        # self.assertEqual(result, 10.0)

        # # Test 4: Test the volume of an object with a quadratically increasing cross-sectional area
        # arr = np.array([[0, 0], [1, 1], [2, 4], [3, 9], [4, 16]])
        # result = volume(arr)
        # self.assertEqual(result, 30.0)

        # # Test 5: Test the volume of an object with a quadratically decreasing cross-sectional area
        # arr = np.array([[0, 16], [1, 9], [2, 4], [3, 1], [4, 0]])
        # result = volume(arr)
        # self.assertEqual(result, 30.0)

if __name__ == '__main__':
    unittest.main()
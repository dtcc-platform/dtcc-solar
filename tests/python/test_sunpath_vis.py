import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dtcc_solar.sun_utils as su

from pvlib import solarposition
from typing import Dict, List, Any
from dtcc_solar.sunpath import Sunpath 


class TestSunpathVisual:

    lat: float
    lon: float
    origin: np.ndarray
    radius: float
    sunpath: Sunpath
    horizon_z: float

    def setup_method(self):
        self.lon = -0.12
        self.lat = 51.5
        self.radius = 5
        self.horizon_z = 0.0
        self.origin = np.array([0,0,0])
        self.sunpath = Sunpath(self.lat, self.lon, self.radius, self.origin)

    def test_analemmas(self):
        year = 2018
        [x, y, z, pos_dict] = self.sunpath.get_analemmas(year, 5)
        ax = su.initialise_plot(self.radius, "Analemmas")
        su.plot_analemmas(pos_dict, self.radius, ax, True, 'autumn_r', 0)
        plt.show()
        pass 

    def test_daypath(self):
        dates = pd.date_range(start = '2019-01-01', end = '2019-09-30', freq = '10D')
        [x, y, z] = self.sunpath.get_daypaths(dates, 10)    
        ax = su.initialise_plot(self.radius, "Day paths")
        su.plot_daypath(x, y, z, self.radius, ax, True) 
        plt.show()
        pass

    def test_single_sun_pos(self):
        time = pd.to_datetime(['2019-05-30 12:20:00'])
        ax = su.initialise_plot(self.radius, "Single sun")
        sun_pos = self.sunpath.get_single_sun(time)
        print(sun_pos)
        su.plot_single_sun(sun_pos[0][0], sun_pos[0][1], sun_pos[0][2], self.radius, ax)
        plt.show()
        pass

    def test_multiple_sun_pos(self):
        dates = pd.to_datetime(['2019-02-21 12:20:00', '2019-06-21 12:20:00', '2019-12-21 12:20:00'])
        ax = su.initialise_plot(self.radius, "Multiple suns")
        sun_positions = self.sunpath.get_multiple_suns(dates)    
        su.plot_multiple_suns(sun_positions, self.radius, ax, True)
        plt.show()
        pass
        
   

if __name__ == "__main__":

    os.system('clear')
    
    test = TestSunpathVisual()

    test.setup_method()

    #test.test_single_sun_pos()
    #test.test_multiple_sun_pos()
    #test.test_analemmas()
    test.test_daypath()



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any
from dtcc_solar.sunpath import Sunpath
from sandbox.sunpath_vis import SunpathVis
from dtcc_solar.utils import SolarParameters


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
        self.origin = np.array([0, 0, 0])
        self.sunpath = Sunpath(self.lat, self.lon, self.radius)
        self.sunvis = SunpathVis()
        self.p = SolarParameters()

    def test_analemmas(self):
        year = 2018
        sun_pos_dict = self.sunpath.get_analemmas(year, 5)
        ax = self.sunvis.initialise_plot(self.radius, "Analemmas")
        self.sunvis.plot_analemmas(sun_pos_dict, self.radius, ax, True, "autumn_r", 0)
        plt.show()
        pass

    def test_daypath(self):
        dates = pd.date_range(start="2019-01-01", end="2019-09-30", freq="10D")
        sun_pos_dict = self.sunpath.get_daypaths(dates, 10)
        ax = self.sunvis.initialise_plot(self.radius, "Day paths")
        self.sunvis.plot_daypath(sun_pos_dict, self.radius, ax, True)
        plt.show()
        pass

    def test_single_sun_pos(self):
        self.p.start_date = "2019-05-30 12:20:00"
        self.p.end_date = "2019-05-30 12:20:00"
        suns = self.sunpath.create_suns(self.p)
        if len(suns) > 0:
            pos = suns[0].position
            ax = self.sunvis.initialise_plot(self.radius, "Single sun")
            self.sunvis.plot_single_sun(pos.x, pos.y, pos.z, self.radius, ax)
            plt.show()
        pass

    def test_multiple_sun_pos(self):
        self.p.start_date = "2019-02-21 12:20:00"
        self.p.end_date = "2019-02-22 12:20:00"
        suns = self.sunpath.create_suns(self.p)
        suns_pos = []
        for sun in suns:
            suns_pos.append([sun.position.x, sun.position.y, sun.position.z])

        suns_pos = np.array(suns_pos)
        ax = self.sunvis.initialise_plot(self.radius, "Multiple suns")
        self.sunvis.plot_multiple_suns(suns_pos, self.radius, ax, True)
        plt.show()
        pass


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Visual sunpath test started -----------------------")

    test = TestSunpathVisual()

    test.setup_method()

    # test.test_single_sun_pos()
    # test.test_multiple_sun_pos()
    # test.test_analemmas()
    test.test_daypath()

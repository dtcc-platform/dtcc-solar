import os
import pandas as pd
from dtcc_solar import smhi_data
from dtcc_solar import meteo_data
from pprint import pp
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import AnalysisType
from dtcc_solar.data_io import Parameters
from dtcc_solar.scripts.main import get_sun_and_sky


class TestOpenMeteoApi:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812
        self.w_file_clm = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.w_file_epw = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
        self.file_name = '../data/models/CitySurfaceS.stl'


    def test_weather_data(self):

        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"
        sunpath = Sunpath(self.lat, self.lon, 1.0)
        a_type = AnalysisType.sun_raycasting
        
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 1, 1, 
                       False, start_date, start_date, end_date, self.w_file_clm, 2)
        
        suns = get_sun_and_sky(p, sunpath)
        suns = meteo_data.get_data_from_api_call(self.lon, self.lat, suns)

        assert suns


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Open Meteo AIP test started -----------------------")

    test = TestOpenMeteoApi()
    test.setup_method()
    test.test_weather_data()








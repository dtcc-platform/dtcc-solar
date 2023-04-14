import os
import pandas as pd
from dtcc_solar import data_io
from dtcc_solar import clm_data
from pprint import pp

class TestClmData:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = -0.12
        self.lat = 51.5
        self.filepath = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"

    def test_weather_data(self):
        time_from = pd.to_datetime("2019-03-01 00:00:00")
        time_to = pd.to_datetime("2019-03-25 23:00:00")
        [w_data_clm, dict_keys_clm] = clm_data.import_weather_data_clm(time_from, time_to, self.filepath)
        assert w_data_clm


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- CLM data test started -----------------------")

    test = TestClmData()
    test.setup_method()
    test.test_weather_data()

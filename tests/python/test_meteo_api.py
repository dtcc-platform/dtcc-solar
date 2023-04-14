import os
import pandas as pd
from dtcc_solar import smhi_data
from dtcc_solar import meteo_data
from pprint import pp

class TestOpenMeteoApi:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812

    def test_weather_data(self):
        time_from = pd.to_datetime("2019-03-22")
        time_to = pd.to_datetime("2019-04-01")

        [w_data_dict, dict_keys_sub] = meteo_data.get_data_from_api_call(self.lon, self.lat, time_from, time_to)

        pp(w_data_dict)
        pp(dict_keys_sub)

        assert w_data_dict


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Open Meteo AIP test started -----------------------")

    test = TestOpenMeteoApi()
    test.setup_method()
    test.test_weather_data()








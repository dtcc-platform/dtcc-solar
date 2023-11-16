import os
import pandas as pd
from pprint import pp
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SolarParameters, DataSource


class TestOpenMeteoApi:
    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812
        self.w_file_clm = (
            "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        )
        self.w_file_epw = (
            "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
        )
        self.file_name = "../data/models/CitySurfaceS.stl"

    def test_weather_data(self):
        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"

        p = SolarParameters(
            file_name=self.file_name,
            weather_file=self.w_file_clm,
            latitude=self.lat,
            longitude=self.lon,
            data_source=DataSource.smhi,
            start_date=start_date,
            end_date=end_date,
            display=False,
            sun_analysis=True,
            sky_analysis=False,
        )

        sunpath = Sunpath(p, 1.0)

        assert sunpath.sunc


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Open Meteo AIP test started -----------------------")

    test = TestOpenMeteoApi()
    test.setup_method()
    test.test_weather_data()

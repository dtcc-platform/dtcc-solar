import os
import pandas as pd
from pprint import pp
from dtcc_solar import data_smhi
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import AnalysisType, SolarParameters, DataSource, ColorBy


class TestSmhiApi:
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
            a_type=AnalysisType.sun_raycasting,
            latitude=self.lat,
            longitude=self.lon,
            display=False,
            data_source=DataSource.smhi,
            color_by=ColorBy.face_sun_angle,
            export=False,
            start_date=start_date,
            end_date=end_date,
        )

        sunpath = Sunpath(p, 1.0)

        assert sunpath.suns

    def test_summer_time(self):
        assert self.assert_no_summer_time_in_data()

    def test_weather_stations_location_import(self):
        pos_dict = data_smhi.get_shmi_stations_from_api()
        pp(pos_dict)
        assert pos_dict

    def assert_no_summer_time_in_data(self):
        # In order to test the the SMHI data does not acount for summer time shift, (which
        # need to be sycronised with the solar position calulations), the hour 02:00 should
        # present in the weather data in the last sunday of march and in the last sunday of
        # october. This structure is based on an EU regulation but may not be followed by
        # countries outside of EU. However, the SMHI Open data API only covers the north of
        # Europe so the check is still relevant for this application.

        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"

        p = SolarParameters(
            file_name=self.file_name,
            weather_file=self.w_file_clm,
            a_type=AnalysisType.sun_raycasting,
            latitude=self.lat,
            longitude=self.lon,
            data_source=DataSource.smhi,
            color_by=ColorBy.face_sun_angle,
            start_date=start_date,
            end_date=end_date,
        )

        sunpath = Sunpath(p, 1.0)

        # Checking that 02:00:00 existing in the transion from winter to summer.
        dst_test_from = [
            "2018-03-25 00:00:00",
            "2018-10-28 00:00:00",
            "2019-03-31 00:00:00",
            "2019-10-27 00:00:00",
            "2020-03-29 00:00:00",
            "2020-10-25 00:00:00",
            "2021-03-28 00:00:00",
            "2021-10-31 00:00:00",
            "2022-03-27 00:00:00",
            "2022-10-30 00:00:00",
        ]

        dst_test_to = [
            "2018-03-25 04:00:00",
            "2018-10-28 04:00:00",
            "2019-03-31 04:00:00",
            "2019-10-27 04:00:00",
            "2020-03-29 04:00:00",
            "2020-10-25 04:00:00",
            "2021-03-28 04:00:00",
            "2021-10-31 04:00:00",
            "2022-03-27 04:00:00",
            "2022-10-30 04:00:00",
        ]

        results = []
        for i in range(0, len(dst_test_from)):
            p.start_date = dst_test_from[i]
            p.end_date = dst_test_to[i]
            suns = sunpath.suns
            for sun in suns:
                hour = sun.datetime_ts.hour
                if hour == 2:
                    results.append(True)
                    break

        if all(results):
            return True

        return False


if __name__ == "__main__":
    os.system("clear")
    print(
        "--------------------- SMHI Open Data API test started -----------------------"
    )

    test = TestSmhiApi()
    test.setup_method()
    # test.test_weather_data()
    # test.test_summer_time()
    test.test_weather_stations_location_import()

import os
import pandas as pd
from dtcc_solar import smhi_data

class TestSmhiApi:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812

    def test_weather_data(self):
        time_from = pd.to_datetime("2019-03-22")
        time_to = pd.to_datetime("2019-04-01")
        w_data_dict = smhi_data.get_data_from_api_call(self.lon, self.lat, time_from, time_to)
        assert w_data_dict

    def test_summer_time(self):
        assert self.assert_no_summer_time_in_data()

    def test_weather_stations_location_import(self):
        pos_dict = smhi_data.get_shmi_stations_from_api()
        print(pos_dict)
        assert pos_dict

    def assert_no_summer_time_in_data(self):

        # In order to test the the SMHI data does not acount for summer time shift, (which 
        # need to be sycronised with the solar position calulations), the hour 02:00 should
        # present in the weather data in the last sunday of march and in the last sunday of
        # october. This structure is based on an EU regulation but may not be followed by 
        # countries outside of EU. However, the SMHI Open data API only covers the north of 
        # Europe so the check is still relevant for this application.    

        # Checking that 02:00:00 existing in the transion from winter to summer.
        dst_test_from = ["2018-03-25 00:00:00", "2018-10-28 00:00:00", 
                         "2019-03-31 00:00:00", "2019-10-27 00:00:00", 
                         "2020-03-29 00:00:00", "2020-10-25 00:00:00",
                         "2021-03-28 00:00:00", "2021-10-31 00:00:00",
                         "2022-03-27 00:00:00", "2022-10-30 00:00:00"]
    
        dst_test_to =   ["2018-03-25 04:00:00", "2018-10-28 04:00:00", 
                         "2019-03-31 04:00:00", "2019-10-27 04:00:00", 
                         "2020-03-29 04:00:00", "2020-10-25 04:00:00",
                         "2021-03-28 04:00:00", "2021-10-31 04:00:00",
                         "2022-03-27 04:00:00", "2022-10-30 04:00:00"]
    
        results = []
        for i in range(0, len(dst_test_from)):
            time_from = dst_test_from[i]
            time_to =   dst_test_to[i]
            [w_data_dict, dict_keys] = smhi_data.get_data_from_api_call(self.lon, self.lat, time_from, time_to)
            for key in w_data_dict:
                parts = key.split('T')
                part2 = parts[1]
                hour = part2[0:2]
                if(hour == "02"):
                    results.append(True)
                    break
            
        if(all(results)):                
            return True
    
        return False


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- SMHI Open Data API test started -----------------------")

    test = TestSmhiApi()
    test.setup_method()
    test.test_weather_data()








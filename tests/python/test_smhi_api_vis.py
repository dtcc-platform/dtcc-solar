import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pp
from dtcc_solar import smhi_data
from dtcc_solar import meteo_data
import dtcc_solar.sun_utils as su

class TestSmhiApiVisual:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812

    def test_smhi_api_weather_data(self):

        #ax = su.initialise_plot_2D(0, 24, 0, 1000, "Normal irradiance")
        time_from = pd.to_datetime("2020-01-28")
        time_to = pd.to_datetime("2020-08-28")
        [w_data_dict, dict_keys] = smhi_data.get_data_from_api_call(self.lon, self.lat, time_from, time_to)
        [w_data_normal, w_data_horizontal] = format_smhi_data_per_day(w_data_dict, dict_keys)
        
        x_data = np.array(range(0,24))

        plt.figure()
        ax_1 = plt.subplot(2,2,1)
        plt.title("Normal irradiance")
        ax_1.set_xlabel('Time of day')
        ax_1.set_ylabel('W/m2')
        ax_1.set_xlim(0, 23)
        ax_1.set_ylim(0, 1000)
        ax_1.set_xticks(range(0,23)) 
        for key in w_data_horizontal:
            y_data_normal = w_data_normal[key]
            plt.plot(x_data,y_data_normal)
        
        ax_2= plt.subplot(2,2,2)
        plt.title("Horizontal irradiance")
        ax_2.set_xlabel('Time of day')
        ax_2.set_ylabel('W/m2')
        ax_2.set_xlim(0, 23)
        ax_2.set_ylim(0, 1000)
        ax_2.set_xticks(range(0,23))
        for key in w_data_horizontal:
            y_data_horizontal = w_data_horizontal[key]
            plt.plot(x_data,y_data_horizontal)
                    
        plt.show()
        pass


def format_smhi_data_per_day(w_data_dict, dict_keys):

    # Collect the data per day instead of per hour. Hence the use of date_keys
    date_keys = np.array([])
    for key in w_data_dict:
        date = key[0:10]
        date_keys = np.append(date_keys, date)

    normal_irr = dict.fromkeys(date_keys)
    horizon_irr = dict.fromkeys(date_keys)
        
    data_ni = []
    data_hi = []
    for key in w_data_dict:
        date_key = key[0:10]
        hour = int(key[11:13])
        data_ni.append(w_data_dict[key]['normal_irradiance'])
        data_hi.append(w_data_dict[key]['horizontal_irradiance'])

        if(hour == 23):
            normal_irr[date_key] = data_ni
            horizon_irr[date_key] = data_hi
            data_ni = []
            data_hi = []

    empty_keys = [k for k, v in normal_irr.items() if v == None]
    normal_irr.pop(empty_keys[0])
        
    empty_keys = [k for k, v in horizon_irr.items() if v == None]
    horizon_irr.pop(empty_keys[0])

    return normal_irr, horizon_irr

if __name__ == "__main__":

    test = TestSmhiApiVisual()
    test.setup_method()
    test.test_smhi_api_weather_data()

    pass







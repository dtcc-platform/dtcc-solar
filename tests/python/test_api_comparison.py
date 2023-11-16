import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pp
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SolarParameters, SunCollection, DataSource
from typing import List, Dict, Any
import copy


class TestWeatherDataComparison:
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

        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"

        self.p_smhi = SolarParameters(
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

        self.p_meteo = copy.deepcopy(self.p_smhi)
        self.p_meteo.data_source = DataSource.meteo

        self.p_clm = copy.deepcopy(self.p_smhi)
        self.p_clm.data_source = DataSource.clm

        self.p_epw = copy.deepcopy(self.p_smhi)
        self.p_epw.weather_file = self.w_file_epw
        self.p_epw.data_source = DataSource.epw

    def test_compare_dict_keys(self):
        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-05 00:00:00"

        self.p_smhi.start_date = start_date
        self.p_smhi.end_date = end_date
        self.p_meteo.start_date = start_date
        self.p_meteo.end_date = end_date
        self.p_clm.start_date = start_date
        self.p_clm.end_date = end_date
        self.p_epw.start_date = start_date
        self.p_epw.end_date = end_date

        sunpath_smhi = Sunpath(self.p_smhi, 1.0, True)
        sunpath_meteo = Sunpath(self.p_meteo, 1.0, True)
        sunpath_clm = Sunpath(self.p_clm, 1.0, True)
        sunpath_epw = Sunpath(self.p_epw, 1.0, True)

        if (
            sunpath_smhi.sunc.count != sunpath_meteo.sunc.count
            or sunpath_smhi.sunc.count != sunpath_clm.sunc.count
            or sunpath_smhi.sunc.count != sunpath_epw.sunc.count
        ):
            print("Keys missmatch")
            return False

        for i in range(sunpath_smhi.sunc.count):
            if (
                sunpath_smhi.sunc.datetime_strs[i]
                != sunpath_meteo.sunc.datetime_strs[i]
                or sunpath_smhi.sunc.datetime_strs[i]
                != sunpath_clm.sunc.datetime_strs[i]
                or sunpath_smhi.sunc.datetime_strs[i]
                != sunpath_epw.sunc.datetime_strs[i]
            ):
                print("Keys missmatch")
                return False

        print("Keys matching asserted")
        assert True

    def test_calculate_monthly_average_data(self):
        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"

        self.p_smhi.start_date = start_date
        self.p_smhi.end_date = end_date
        self.p_meteo.start_date = start_date
        self.p_meteo.end_date = end_date
        self.p_clm.start_date = start_date
        self.p_clm.end_date = end_date
        self.p_epw.start_date = start_date
        self.p_epw.end_date = end_date

        sunpath_smhi = Sunpath(self.p_smhi, 1.0, True)
        sunpath_meteo = Sunpath(self.p_meteo, 1.0, True)
        sunpath_clm = Sunpath(self.p_clm, 1.0, True)
        sunpath_epw = Sunpath(self.p_epw, 1.0, True)

        # pp(sunpath_smhi.sunc.datetime_strs)
        # pp(sunpath_smhi.sunc.time_stamps)

        [w_data_normal_smhi, w_data_horizontal_smhi] = format_data_per_day_suns(
            sunpath_smhi.sunc
        )

        smhi_normal_avrg = get_monthly_average_data(w_data_normal_smhi)
        smhi_horizon_avrg = get_monthly_average_data(w_data_horizontal_smhi)

        [w_data_normal_meteo, w_data_horizontal_meteo] = format_data_per_day_suns(
            sunpath_meteo.sunc
        )

        meteo_normal_avrg = get_monthly_average_data(w_data_normal_meteo)
        meteo_horizon_avrg = get_monthly_average_data(w_data_horizontal_meteo)

        [w_data_normal_clm, w_data_horizontal_clm] = format_data_per_day_suns(
            sunpath_clm.sunc
        )

        clm_normal_avrg = get_monthly_average_data(w_data_normal_clm)
        clm_horizon_avrg = get_monthly_average_data(w_data_horizontal_clm)

        [w_data_normal_epw, w_data_horizontal_epw] = format_data_per_day_suns(
            sunpath_epw.sunc
        )

        epw_normal_avrg = get_monthly_average_data(w_data_normal_epw)
        epw_horizon_avrg = get_monthly_average_data(w_data_horizontal_epw)

        plt.figure(1)
        ax_1 = plt.subplot(2, 4, 1)
        title = "SMHI Monthly average normal irradiance"
        plot_montly_average(
            title, ax_1, smhi_normal_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_2 = plt.subplot(2, 4, 5)
        title = "SMHI Monthly average horisontal irradiance"
        plot_montly_average(
            title, ax_2, smhi_horizon_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_3 = plt.subplot(2, 4, 2)
        title = "Meteo Monthly average normal irradiance"
        plot_montly_average(
            title, ax_3, meteo_normal_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_4 = plt.subplot(2, 4, 6)
        title = "Meteo Monthly average horisontal irradiance"
        plot_montly_average(
            title, ax_4, meteo_horizon_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_5 = plt.subplot(2, 4, 3)
        title = "CLM Monthly average normal irradiance"
        plot_montly_average(
            title, ax_5, clm_normal_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_6 = plt.subplot(2, 4, 7)
        title = "CLM Monthly average horisontal irradiance"
        plot_montly_average(
            title, ax_6, clm_horizon_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_7 = plt.subplot(2, 4, 4)
        title = "EPW Monthly average normal irradiance"
        plot_montly_average(
            title, ax_7, epw_normal_avrg, "Time of day", "W/m2", 0, 1000
        )

        ax_8 = plt.subplot(2, 4, 8)
        title = "EPW Monthly average horisontal irradiance"
        plot_montly_average(
            title, ax_8, epw_horizon_avrg, "Time of day", "W/m2", 0, 1000
        )

        plt.show()

    def test_compare_weather_data(self):
        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"

        self.p_smhi.start_date = start_date
        self.p_smhi.end_date = end_date
        self.p_meteo.start_date = start_date
        self.p_meteo.end_date = end_date
        self.p_clm.start_date = start_date
        self.p_clm.end_date = end_date
        self.p_epw.start_date = start_date
        self.p_epw.end_date = end_date

        sunpath_smhi = Sunpath(self.p_smhi, 1.0, True)
        sunpath_meteo = Sunpath(self.p_meteo, 1.0, True)
        sunpath_clm = Sunpath(self.p_clm, 1.0, True)
        sunpath_epw = Sunpath(self.p_epw, 1.0, True)

        [w_data_normal_smhi, w_data_horizontal_smhi] = format_data_per_day_suns(
            sunpath_smhi.sunc
        )
        [w_data_normal_meteo, w_data_horizontal_meteo] = format_data_per_day_suns(
            sunpath_meteo.sunc
        )
        [w_data_normal_clm, w_data_horizontal_clm] = format_data_per_day_suns(
            sunpath_clm.sunc
        )
        [w_data_normal_epw, w_data_horizontal_epw] = format_data_per_day_suns(
            sunpath_epw.sunc
        )

        if sunpath_smhi.sunc.count != sunpath_meteo.sunc.count:
            print("SMHI and Open Meteo data format missmatch")
            return None

        log_normal = w_data_normal_smhi.copy()
        log_horizontal = w_data_normal_smhi.copy()

        np.seterr(divide="ignore")

        for key in w_data_normal_smhi:
            a = np.array(w_data_normal_smhi[key])
            b = np.array(w_data_normal_meteo[key])
            log_n = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            log_n = np.log(log_n)
            log_normal[key] = log_n

            a = np.array(w_data_horizontal_smhi[key])
            b = np.array(w_data_horizontal_meteo[key])
            log_h = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            log_h = np.log(log_h)
            log_horizontal[key] = log_h

        np.seterr(divide="warn")

        x_data = np.array(range(0, 24))

        print("Weather data computation asserted")

        plt.figure(1)
        ax_1 = plt.subplot(2, 4, 1)
        plt.title("SMHI Normal irradiance")
        subplot_setup(ax_1, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_smhi:
            y_data_normal = w_data_normal_smhi[key]
            plt.plot(x_data, y_data_normal)

        ax_2 = plt.subplot(2, 4, 5)
        plt.title("SMHI Horizontal irradiance")
        subplot_setup(ax_2, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_smhi:
            y_data_horizontal = w_data_horizontal_smhi[key]
            plt.plot(x_data, y_data_horizontal)

        ax_3 = plt.subplot(2, 4, 2)
        plt.title("Open Meteo Normal irradiance")
        subplot_setup(ax_3, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_meteo[key]
            plt.plot(x_data, y_data_normal)

        ax_4 = plt.subplot(2, 4, 6)
        plt.title("Open Meteo Horizontal irradiance")
        subplot_setup(ax_4, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_meteo[key]
            plt.plot(x_data, y_data_horizontal)

        ax_5 = plt.subplot(2, 4, 3)
        plt.title("CLM-file Normal irradiance")
        subplot_setup(ax_5, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_clm[key]
            plt.plot(x_data, y_data_normal)

        ax_6 = plt.subplot(2, 4, 7)
        plt.title("CLM-file Horizontal irradiance")
        subplot_setup(ax_6, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_clm[key]
            plt.plot(x_data, y_data_horizontal)

        ax_7 = plt.subplot(2, 4, 4)
        plt.title("EPW-file Normal irradiance")
        subplot_setup(ax_7, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_epw[key]
            plt.plot(x_data, y_data_normal)

        ax_8 = plt.subplot(2, 4, 8)
        plt.title("EPW-file Horizontal irradiance")
        subplot_setup(ax_8, "Time of day", "W/m2", 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_epw[key]
            plt.plot(x_data, y_data_horizontal)

        plt.figure(2)
        ax_10 = plt.subplot(1, 2, 1)
        plt.title("Normal irradinance - Log(SMHI/Meteo)")
        ax_10.set_xlabel("Time of day")
        ax_10.set_xticks(range(0, 23))
        for key in w_data_horizontal_meteo:
            y_data_normal = log_normal[key]
            plt.plot(x_data, y_data_normal)

        ax_11 = plt.subplot(1, 2, 2)
        plt.title("Horizontal irradiance - Log(SMHI/Meteo)")
        ax_11.set_xlabel("Time of day")
        ax_11.set_xticks(range(0, 23))
        for key in w_data_horizontal_meteo:
            y_data_horizontal = log_horizontal[key]
            plt.plot(x_data, y_data_horizontal)

        plt.show()
        pass


def get_monthly_average_data(data_dict: Dict[str, List[float]]):
    avrg_data = np.zeros((12, 24))
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    key_list = list(data_dict.keys())

    for i, key in enumerate(key_list):
        month = get_month_from_dict_key(key)
        month_index = month - 1
        for j in range(0, 23):
            avrg = data_dict[key][j] / days_per_month[month_index]
            avrg_data[month_index, j] += avrg

    return avrg_data


def get_month_from_dict_key(dict_key):
    month = int(dict_key[5:7])
    return month


def plot_montly_average(title, ax, data, x_label, y_label, y_min, y_max):
    x_data = range(0, 24)
    plt.title(title)
    subplot_setup(ax, x_label, y_label, 0, 23, y_min, y_max)
    for i in range(0, 12):
        y_data_normal = data[i, :]
        plt.plot(x_data, y_data_normal)


def subplot_setup(ax, x_label, y_label, x_min, x_max, y_min, y_max):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(0, 23))


def format_data_per_day_suns(sunc: SunCollection):
    date_keys = []
    for datetime_str in sunc.datetime_strs:
        date = datetime_str[0:10]
        date_keys.append(date)

    normal_irr = dict.fromkeys(date_keys)
    horizon_irr = dict.fromkeys(date_keys)
    data_ni = []
    data_hi = []

    for i in range(sunc.count):
        key = sunc.datetime_strs[i]
        date_key = key[0:10]
        hour = sunc.time_stamps[i].hour
        data_ni.append(sunc.irradiance_dn[i])
        data_hi.append(sunc.irradiance_di[i])
        if hour == 23:
            normal_irr[date_key] = data_ni
            horizon_irr[date_key] = data_hi
            data_ni = []
            data_hi = []

    empty_keys = [k for k, v in normal_irr.items() if v == None]
    if len(empty_keys):
        normal_irr.pop(empty_keys[0])

    empty_keys = [k for k, v in horizon_irr.items() if v == None]
    if len(empty_keys):
        horizon_irr.pop(empty_keys[0])

    return normal_irr, horizon_irr


# To avoid error message "RuntimeWarning: divide by zero encountered in np.log(log_h)"
def safe_log10(x, eps=1e-10):
    result = np.where(x > eps, x, -10)
    np.log10(result, out=result, where=result > 0)
    return result


if __name__ == "__main__":
    os.system("clear")

    print("--------------------- API Comparision test started -----------------------")

    test = TestWeatherDataComparison()
    test.setup_method()
    # test.test_compare_dict_keys()
    test.test_calculate_monthly_average_data()
    # test.test_compare_weather_data()

    pass

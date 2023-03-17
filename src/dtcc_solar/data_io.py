import numpy as np
import pandas as pd
import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

from dtcc_solar.utils import AnalysisType, ColorBy


class Parameters:
    def __init__(self, a_type, fileName, lat, lon, prep_disp, disp, origin, color_by, export, one_date, s_date, e_date):
        self.a_type = AnalysisType(a_type)
        self.file_name = fileName
        self.latitude = lat
        self.longitude = lon 
        self.prepare_display = bool(prep_disp)
        self.display = bool(disp)
        self.origin = np.array(origin, dtype= float)
        self.color_by = ColorBy(color_by)
        self.export = export
        self.one_date = one_date
        self.start_date = s_date
        self.end_date = e_date
        self.discretisation = '1H' 

def export_results(solpos):
    with open("sunpath.txt", "w") as f:
        
        for item in solpos['zenith'].values:
            f.write(str(item[0]) + '\n')

def print_list(listToPrint, path):
    counter = 0
    with open(path, 'w') as f:
        for row in listToPrint:
            f.write( str(row) + '\n')
            counter += 1 

    print("Export completed")


def print_dict(dictToPrint, filename):
    counter = 0
    with open(filename, "w") as f:
        for key in dictToPrint:
            f.write('Key:' + str(key) + ' ' + str(dictToPrint[key])+'\n')


def print_results(shouldPrint,faceRayFaces):
    counter = 0
    if shouldPrint:
        with open("faceRayFace.txt", "w") as f:
            for key in faceRayFaces:
                f.write('Face index:' + str(key) + ' ' + str(faceRayFaces[key])+'\n')
                counter += 1 

    print(counter)


def import_weather_date_epw(p:Parameters):
    file = '/Users/jensolsson/Documents/Dev/DTCC/CitySolar/weatherdata/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw' 
    year_dict_keys = []
    direct_normal_radiation = []
    diffuse_horisontal_radiation = []
    year_weather_data = dict()
    search = False

    with open(file, 'r') as f:
        for line in f:
            if search:
                line_segs = line.split(',')
                year = line_segs[0]
                month = make_double_digit_str(line_segs[1])
                day = make_double_digit_str(line_segs[2])
                hour = make_double_digit_str(line_segs[3])
                date_key = year + '-' + month + '-' + day + ' ' + hour + ':00:00'
                year_dict_keys.append(date_key)
                direct_normal_radiation.append(float(line_segs[14]))  
                diffuse_horisontal_radiation.append(float(line_segs[15]))
                
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True    

    for i, key in enumerate(year_dict_keys):
        d_type = ['diffuse_radiation', 'direct_normal_radiation']
        a_dict = dict.fromkeys([d for d in d_type])
        a_dict['diffuse_radiation'] = diffuse_horisontal_radiation[i]
        a_dict['direct_normal_radiation'] = direct_normal_radiation[i] 
        year_weather_data[key] = a_dict
        

    print_dict(year_weather_data, "w_data_year_epw.txt")

def make_double_digit_str(s:str):
    if len(s) == 1:
        s = '0' + s
    return s     

def find_date_range_epw(file):
    start_year = 2200
    end_year = 1800
    search = False
    with open(file, 'r') as f:
        for line in f:
            if search:
                year = int(line[0:4]) 
                if year < start_year:
                    start_year = year
                if year > end_year:
                    end_year = year
                    
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True       

    start_date = str(start_year) + '-01-01 00:00:00'
    end_date = str(end_year) + '-12-31 23:59:00'

    return start_date, end_date


def import_weather_data_clm(p:Parameters):
    
    line_index = 0
    data_counter = 0
    #The data file contains weather data for each hour during the year 2015.
    file = '/Users/jensolsson/Documents/Dev/DTCC/CitySolar/weatherdata/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'
    year_dates = pd.date_range(start = '2015-01-01 00:00:00', end = '2015-12-31 23:59:00', freq = '1H')
    year_dict_keys = np.array([str(d) for d in year_dates])
    year_weather_data = dict.fromkeys(year_dict_keys)

    with open(file, 'r') as f:
        for line in f:
            #The first 12 lines contains information of the data structure
            if line_index > 12 and line[0] != '*':
               year_weather_data[str(year_dates[data_counter])] = line2numbers(line)     
               data_counter += 1     
            
            line_index += 1

    print_dict(year_weather_data, "w_data_year.txt")
    
    if( p.a_type == AnalysisType.sun_raycasting or 
        p.a_type == AnalysisType.sky_raycasting or
        p.a_type == AnalysisType.sky_raycasting_some):
        sub_dates = pd.date_range(start = p.one_date, end = p.one_date, freq = '1H')
    else:
        sub_dates = pd.date_range(start = p.start_date, end = p.end_date, freq = '1H')

    [sub_weather_data, sub_dict_keys] = get_weather_data_subset(year_weather_data, sub_dates)    

    print_dict(sub_weather_data, "w_data_sub.txt")

    return sub_weather_data, sub_dict_keys, sub_dates

def get_weather_data_subset(year_weather_data, sub_dates):
    sub_dict_keys = np.array([str(d) for d in sub_dates])
    sub_weather_data = dict.fromkeys(sub_dict_keys)

    for key in sub_dict_keys:
        sub_weather_data[key] = year_weather_data[key]

    return sub_weather_data, sub_dict_keys


def line2numbers(line):
    keys = ['diffuse_radiation', 'direct_normal_radiation']
    adict = dict.fromkeys([k for k in keys])
    line = line.replace('\n', '')
    linesegs = line.split(',')
    if(len(linesegs) == 6):
        adict['diffuse_radiation'] = float(linesegs[0])
        adict['direct_normal_radiation'] = float(linesegs[2])

    return adict
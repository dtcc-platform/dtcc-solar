import numpy as np
import math
from pvlib import solarposition
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class DataSource(Enum):
    smhi = 1
    meteo = 2
    clm = 3
    epw = 4

class ColorBy(Enum):
    face_sun_angle = 1
    face_sun_angle_shadows = 2
    face_irradiance = 3
    face_shadows = 4
    face_in_sky = 5
    face_diffusion = 6

class Mode(Enum):
    single_sun = 1
    multiple_sun = 2

class AnalysisType(Enum):
    sun_raycasting = 1
    sky_raycasting = 2
    com_raycasting = 3
    sky_raycasting_some = 4         # Only for debugging.

class AnalysisTypeDev(Enum):
    vertex_raycasting = 1
    subdee_shading = 2
    diffuse_dome = 3
    diffuse_some = 4
    diffuse_all = 5        

class Analyse(Enum):
    Time = 1
    Day = 2
    Year = 3
    Times = 4

class Shading(Enum):
    Sun = 1
    Boarder = 2
    Shade = 3
    

@dataclass
class Vec3:
    x: float
    y: float
    z: float

@dataclass
class Sun:
    datetime_str: str                   # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_ts: pd.Timestamp           # TimeStamp object with the same date and time
    irradiance_dn: float = 0.0          # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth  
    irradiance_hi: float = 0.0          # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dh: float = 0.0          # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    over_horizon: bool = False          # True if the possition of over the horizon, otherwise false.
    zenith: float = 0.0                 # Angle between earth surface normal and the reversed solar vector (both pointing away for the earth surface)
    position: Vec3 = Vec3(0,0,0)        # Position of the  sun in cartesian coordinates based on the size of the model
    sun_vec: Vec3 = Vec3(0,0,0)         # Normalised solar vector for calculations

    
def convert_vec3_to_ndarray(vec: Vec3):
    return np.array([vec.x, vec.y, vec.z])

def create_list_of_vectors(x_list, y_list, z_list) -> List[Vec3]:
    vector_list = []
    for i in range(0, len(x_list)):
        vec = Vec3(x = x_list[i], y = y_list[i], z = z_list[i])
        vector_list.append(vec)
    return vector_list  

def create_suns(start_date: str, end_date: str):
    time_from = pd.to_datetime(start_date)
    time_to = pd.to_datetime(end_date)
    suns = []
    times = pd.date_range(start = time_from, end = time_to, freq = 'H')
    for time in times:
        sun = Sun(str(time), time)
        suns.append(sun)
        
    return suns  

def colorFader(mix): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb('blue'))
    c2=np.array(mpl.colors.to_rgb('red'))
    print(c1)
    print(c2)
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def GetSunVecFromSunPos(sunPos, origin):
    sunVec = origin - sunPos
    sunVecNorm = normalise_vector(sunVec)
    return sunVecNorm  

def get_sun_vecs_from_sun_pos(sunPosList, origin):
    sunVecs = []
    for i in range(0, len(sunPosList)):
        sunPos = sunPosList[i]
        sunVec = origin - sunPos
        sunVecNorm = normalise_vector(sunVec)
        sunVecs.append(sunVecNorm)
    return sunVecs  

def get_dict_keys_from_suns_datetime(suns: List[Sun]):
    dates = []
    for sun in suns:
        if sun.over_horizon:
            dates.append(sun.datetime_str)
    return dates    
 
def get_sun_vecs_dict_from_sun_pos(sunPosList, origin, dict_keys):
    sunVecs = dict.fromkeys(dict_keys)
    counter = 0
    for key in dict_keys:
        sunPos = sunPosList[counter]
        sunVec = origin - sunPos
        sunVecNorm = normalise_vector(sunVec)
        sunVecs[key] = sunVecNorm
        counter += 1
    return sunVecs  

def VectorFromPoints(pt1, pt2):
    vec = [pt2[0] - pt1[0], pt2[1] - pt1[1], pt2[2] - pt1[2]]
    return vec

def normalise_vector(vec):
    length = calc_vector_length(vec)
    vecNorm = np.zeros(3)
    vecNorm = vec / length
    return vecNorm

def normalise_vector3(vec: Vec3):
    length = calc_vector_length3(vec)
    vecNorm = Vec3(x = (vec.x / length), y = (vec.y / length), z = (vec.z / length))
    return vecNorm

def reverse_vector(vec):    
    vecRev = -1.0 * vec
    return vecRev

def calc_vector_length(vec):
    length = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    return length

def calc_vector_length3(vec:Vec3):
    length = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
    return length

def scale_vector3(vec:Vec3, sf:float):
    scaled_vec = Vec3(x = sf * vec.x, y = sf * vec.y, z = sf * vec.z)
    return scaled_vec

def vector_angle(vec1, vec2):
    lengthV1 = calc_vector_length(vec1)
    lengthV2 = calc_vector_length(vec2)
    scalarV1V2 = ScalarProduct(vec1, vec2)
    denominator = lengthV1 * lengthV2
    if denominator != 0:
        angle = math.acos(scalarV1V2 / (denominator))
        return angle
    return 0

def calculate_normal(v1, v2):
    v3 = cross_product(v1, v2)
    normal = normalise_vector(v3)
    return normal

def cross_product(a, b):
    vCross = np.zeros(3)
    vCross[0] = a[1]*b[2] - a[2]*b[1]
    vCross[1] = a[0]*b[2] - a[2]*b[0]
    vCross[2] = a[0]*b[1] - a[1]*b[0]
    return vCross        

def ScalarProduct(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]      

def AvrgVertex(v1,v2,v3):
    x = (v1[0] + v2[0] + v3[0])/3.0
    y = (v1[1] + v2[1] + v3[1])/3.0
    z = (v1[2] + v2[2] + v3[2])/3.0
    return [x,y,z]

def MoveVertex(v, mVec, distance):
    movedVertex = [0, 0, 0]
    movedVertex[0] = v[0] + mVec[0] * distance
    movedVertex[1] = v[1] + mVec[1] * distance
    movedVertex[2] = v[2] + mVec[2] * distance
    return movedVertex 

def IsFaceBlocked(thisTriIndex, intersectedTriangleIndices):
    for iti in intersectedTriangleIndices:
        if iti != thisTriIndex:                 #If there is an intersection which is not the self then the face is blocked.
            return True
    return False        


def create_dict_keys(time_from: pd.Timestamp, time_to: pd.Timestamp) -> np.ndarray:
    
    #Datetimeindex, prb unfit to be used as dictionaty keys 
    dict_keys = pd.date_range(start = time_from, end = time_to, freq = '1H')
    
    #Keys as array of strings with date and time
    dict_keys_str = np.array([str(d) for d in dict_keys])

    #Keys as array of Timestamps
    dict_keys_list = dict_keys.to_list()
    
    return dict_keys_str
 
def Distance(v1, v2):
    return math.sqrt(math.pow((v1[0] - v2[0]),2) + math.pow((v1[1] - v2[1]),2) + + math.pow((v1[2] - v2[2]),2)) 

def GetBlendedColor(percentage):

    if (percentage >= 0.0 and percentage <= 25.0):
        #Blue fading to Cyan [0,x,255], where x is increasing from 0 to 255
        frac = percentage / 25.0
        return [0.0, (frac * 1.0), 1.0, 1.0]
    elif (percentage <= 50.0):
        #Cyan fading to Green [0,255,x], where x is decreasing from 255 to 0
        frac = 1.0 - abs(percentage - 25.0) / 25.0
        return [0.0, 1.0, (frac * 1.0), 1.0]
    elif (percentage <= 75.0):
        #Green fading to Yellow [x,255,0], where x is increasing from 0 to 255
        frac = abs(percentage - 50) / 25
        return [(frac * 1.0), 1.0, 0, 1 ]
    elif (percentage <= 100):
        #Yellow fading to red [255,x,0], where x is decreasing from 255 to 0
        frac = 1 - abs(percentage - 75) / 25.0
        return [1, (frac * 1), 0, 1]

    elif (percentage > 100):
        return [1, 0, 0, 1]

    return [0.5, 0.5, 0.5, 1.0]

def GetBlendedColor(min, max, value):
    diff = max - min
    newMax = diff
    newValue = value - min
    percentage = 100.0 * (newValue / newMax)

    if (percentage >= 0.0 and percentage <= 25.0):
        #Blue fading to Cyan [0,x,255], where x is increasing from 0 to 255
        frac = percentage / 25.0
        return [0.0, (frac * 1.0), 1.0 , 1.0]
    
    elif (percentage > 25.0 and percentage <= 50.0):
        #Cyan fading to Green [0,255,x], where x is decreasing from 255 to 0
        frac = 1.0 - abs(percentage - 25.0) / 25.0
        return [0.0, 1.0, (frac * 1.0), 1.0]
    
    elif (percentage > 50.0 and percentage <= 75.0):
        #Green fading to Yellow [x,255,0], where x is increasing from 0 to 255
        frac = abs(percentage - 50.0) / 25.0
        return [(frac * 1.0), 1.0, 0.0, 1.0 ]

    elif (percentage > 75.0 and percentage <= 100.0):
        #Yellow fading to red [255,x,0], where x is decreasing from 255 to 0
        frac = 1.0 - abs(percentage - 75.0) / 25.0
        return [1.0, (frac * 1.0), 0.0, 1.0]

    elif (percentage > 100.0):
        #Returning red if the value overshoot the limit.
        return [1.0, 0.0, 0.0, 1.0 ]

    return [0.5, 0.5, 0.5, 1.0 ]

def GetBlendedColorRedAndBlue(max, value):
    frac = 0
    if(max > 0):
        frac = value / max
    return [frac, 0.0, 1 - frac, 1.0]

def GetBlendedColorBlackAndWhite(max, value):
    frac = 0
    if(max > 0):
        frac = value / max
    return [frac, frac, frac, 1.0]

def GetBlendedSunColor(max, value):
    percentage = 100.0 * (value / max)
    if (value < 0):
        #Blue fading to Cyan [0,x,255], where x is increasing from 0 to 255
        return [1.0, 1.0, 1.0, 1.0]
    else:
        #Cyan fading to Green [0,255,x], where x is decreasing from 255 to 0
        frac = 1 - percentage/100
        return [1.0, (frac * 1.0), 0.0, 1.0]

def ReverseMask(mask):
    revMask = [not elem for elem in mask]
    return revMask

def CountElementsInDict(dict):     
    count = 0
    for key in dict:
        n = len(dict[key])
        count += n    
    
    return count 

def get_index_of_closest_point(point, array_of_points):
    
    dmin = 10000000000
    index = -1
    for i in range(0, len(array_of_points)):
        d = Distance(point, array_of_points[i])
        if(d < dmin):
            dmin = d
            index = i

    return index        

# Remove subsequent duplicates in the pandas data frame. 
def remove_date_range_duplicates(subset:pd.DataFrame):
    counter = 0
    index_list = subset.index
    for index in index_list:
        current_day = index.day
        
        if(counter > 0):
           if(previous_day == current_day):
               subset = subset.drop(index)
               #print("Dropped item: " + str(index) ) 
        
        previous_day = current_day
        counter += 1

    return subset


def count_elements_in_dict(a_dict: Dict[int, List[Vec3]]):

    counter = 0
    for key in a_dict: 
        counter += len(a_dict[key])

    return counter


# The dict key format is (without blank spaces):
# Year - Month - Day T hour : minute : second
# Example 2018-03-23T12:00:00
def check_dict_keys_format(dict_keys: List[str]):
    for key in dict_keys:
        if not is_dict_key_format_correct(key):
            return False
    return True

# Correct format example 2018-03-23T12:00:00
def is_dict_key_format_correct(dict_key:str):

    if(len(dict_key) < 19):
        return False

    year = dict_key[0:4]
    month = dict_key[5:7]
    day = dict_key[8:10]
    hour = dict_key[11:13]
    min = dict_key[14:16]
    sec = dict_key[17:19]
    
    is_nan = np.all(np.array([int(year), int(month), int(day), int(hour), int(min), int(sec)]))    
    separators = np.array([dict_key[4], dict_key[7], dict_key[10], dict_key[13], dict_key[16]])
    correct_separators = np.array(['-', '-', 'T', ':', ':'])
    res_sep = np.array_equal(separators, correct_separators)

    if not is_nan and res_sep:
        return True 

    return False


def format_dict_keys(dict_keys):
    new_dict_keys = []
    for key in dict_keys:
        new_key = key.replace(' ', 'T')
        new_dict_keys.append(new_key)

    new_dict_keys = np.array(new_dict_keys)
    return new_dict_keys

def get_weather_data_subset(year_weather_data, sub_dict_keys):
    sub_weather_data = dict.fromkeys(sub_dict_keys)
    for key in sub_dict_keys:
        sub_weather_data[key] = year_weather_data[key]

    return sub_weather_data


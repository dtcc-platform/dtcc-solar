import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

import numpy as np
import math
from pvlib import solarposition
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass

class ColorBy(Enum):
    face_sun_angle = 1
    face_sun_angle_shadows = 2
    face_irradiance = 3
    face_shadows = 4
    face_in_sky = 5
    face_diffusion = 6

class AnalysisType(Enum):
    sun_raycasting = 1
    sky_raycasting = 2
    sky_raycasting_some = 3
    sun_raycast_iterative = 4
    com_iterative = 5

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

def create_list_of_vectors(x_list, y_list, z_list):
    vector_list = []
    for i in range(0, len(x_list)):
        vec = Vec3(x = x_list[i], y = y_list[i], z = z_list[i])
        vector_list.append(vec)
    return vector_list    

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

def get_sun_vecs_dict_from_sun_pos(sunPosList, origin, dict_keys):
    sunVecs = dict.fromkeys(dict_keys)
    counter = 0
    print(len(dict_keys))
    print(len(sunPosList))
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
    length = CalcVectorLength(vec)
    vecNorm = np.zeros(3)
    vecNorm = vec / length
    #vecNorm[0] = vec[0]/length
    #vecNorm[1] = vec[1]/length
    #vecNorm[2] = vec[2]/length
    return vecNorm

def reverse_vector(vec):    
    vecRev = -1.0 * vec
    #vecRev = [0,0,0]
    #vecRev[0] = -1.0 * vec[0]
    #vecRev[1] = -1.0 * vec[1]
    #vecRev[2] = -1.0 * vec[2]
    return vecRev

def CalcVectorLength(vec):
    length = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    return length

def VectorAngle(vec1, vec2):
    lengthV1 = CalcVectorLength(vec1)
    lengthV2 = CalcVectorLength(vec2)
    scalarV1V2 = ScalarProduct(vec1, vec2)
    angle = math.acos(scalarV1V2 / (lengthV1 * lengthV2))
    return angle

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
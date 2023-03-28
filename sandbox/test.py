from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.matlib as npmat
import matplotlib as mpl
from pytz import all_timezones
import typing

os.system('clear')

series = pd.Series([11,21,31,41])

print(series)

print(series.values)

print(type(series.values))

arr = np.array([[21,22,23,24],[21,22,23,24]])

print(arr)

print(type(arr))

arr[0,:] = series.values

print(arr)

arr2d = np.zeros((4,6))

print(arr2d)

arr1D = np.array([1,2,8,4,5,6,7,1,9])

mask = (arr1D >= 5)

print(mask)

print(arr1D[2:4])

arr1DsubIndices = np.where(arr1D > 5)

print(arr1DsubIndices)

print(arr1D[arr1DsubIndices])

color=[f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(25)]

#print(color)


C = npmat.repmat([1,2,3],5,1)

print(C)


arrAssem = np.array([arr1D[0], arr1D[1]])

print(arrAssem)

date = pd.to_datetime('2019-03-21 9:00:00')

hours = 12  #hours
stepMin = 5 #minutes
nEvaluation = hours * 60 / stepMin

times = pd.date_range(date, date+pd.Timedelta(str(hours) + 'h'), freq= str(stepMin) + 'min')

x = arr1D[0:len(arr1D):3]

print(x)




a = np.zeros(12, dtype=bool)

print(a)

b = np.zeros([10,3])

b[3,:] = [1,1,1]
b[4] = [2,2,2]

print(b)

a = np.array([1,2,3,4,5,6,7,8,9])

b = np.ones(9, dtype=bool)

c = np.array([3,1,6])

for i in c:
    b[i] = False

d = a[b]

print(d)

aa =  a[a==5]

print(aa)

tf = np.array([True, False, False, True])

arr = np.array([1.3, 4.3, 6.5, 3.4])

#arr = np.append(arr, [9.4])

print(arr[tf])

startDate = str(2019) + '-01-01 12:30:00'
endDate = str(2019) + '-01-02 11:00:00'    
times1 = pd.date_range(start = startDate, end = endDate, freq = 'H')
print(times1.timetz)

tz = 'Europe/Stockholm'

times2 = pd.date_range(start = startDate, end = endDate, freq = 'H', tz=tz)
print(times2.timetz)

print(times1)
print(times2)

print(times1.hour)
print(times2.hour)



arr = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
sun = 5 * np.array([9,2,2])
print(arr + sun)

v1 = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3]])
v2 = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3]])
v3 = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3]])

fMid = (v1 + v2 + v3)/3.0

#print(fMid)

arr = np.array([1.3, 4.3, 6.5, 3.4, 7.5, 2.4, 2.5, 1.4])

print(arr[0:3])


a = np.ones(1)
b = np.zeros(1)
c = np.ones(1)

mat = np.c_[a, b, c]

print(a)

face_data_dict = dict.fromkeys([fi for fi in range(0, 10)])

for i in range(0,10):
    face_data_dict[i] = [0,3,4]

print(face_data_dict.items())

arr = np.array([1.3, 4.3, 6.5, 3.4, 7.5, 2.4, 2.5, 1.4])

#arrb = (arr > 2.4)

#print(int(arrb))
n = 10
max = 2 * np.pi
print(max)

ray_pts = np.zeros((10, 3))
print(ray_pts)

arr = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3],[0, 0, 0]])
print(arr)

pt = np.array([5,0,0])

pts = pt + arr

print(pts)

arr2D = np.array([[[1, 2, 3]],[[1, 5, 3]]])
print(arr)

arr3D = np.array([
                    [[1, 2, 3],[1, 5, 3],[1, 2, 3],[0, 0, 0]],
                    [[11, 22, 13],[31, 25, 13],[21, 32, 43],[50, 60, 30]]
                ])
print(arr3D)

arrAdd = arr2D + arr3D

print(arrAdd)

arr3Dz = np.zeros([1,2,3])

arr3Dz[0,0,:] = [6,6,6]

print(arr3Dz)



arr2Rep = np.array([[1,2,3]])

arrRep = np.repeat(arr2Rep, 5, axis = 0)

print(arrRep)

print('------')
arr1 = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3],[5, 4, 3]])
arr2 = np.array([[3, 1, 2],[4, 3, 2],[6 ,2, 2],[1, 2, 3]])
print(arr1)
print(arr2)

arr3 = np.cross(arr1, arr2)

arr3Len = np.sqrt((arr3 ** 2).sum(-1))[..., np.newaxis]

indices = [0,2]

print(arr3Len[indices])


arr1 = np.array([1, 5, 6])
arr2 = np.array([1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12])
print(arr1)
print(arr2)

arr3 = np.delete(arr2, arr1)

print(arr3)

face_in_sun = np.array([])

print(face_in_sun)


ray_pts = np.zeros(6)

print(ray_pts)


arr1 = np.array([[1, 2, 3],[1, 5, 3],[1, 2, 3],[5, 4, 3]])
arr2 = np.array(['sdvsdv', 'sdvsdv', 'sdvbsdfbvs', 'sdvsdvsdbfd'])
mask = [True, True, False,True]
print(arr1)
print(arr2)
arr1 = arr1[mask]
arr2 = arr2[mask]
print(arr1)
print(arr2)



start_date = '2015-03-30 01:00:00'
end_date = '2015-03-31 00:00:00'

sub_dates = pd.date_range(start = start_date, end = end_date, freq = '1H')

print(sub_dates)


mystr = "0022"

print(mystr[0:2])



loop_pts = dict.fromkeys(range(0,3))

print(loop_pts)

for key in loop_pts:
    loop_pts[key] = "Apa"


def func(a_dict: typing.Dict[int, str]):
    for key in a_dict:
        print(a_dict[key])


func(loop_pts)

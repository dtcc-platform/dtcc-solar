from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.matlib as npmat
import matplotlib as mpl
from pytz import all_timezones
import typing
from pprint import pp

os.system("clear")

series = pd.Series([11, 21, 31, 41])

print(series)

print(series.values)

print(type(series.values))

arr = np.array([[21, 22, 23, 24], [21, 22, 23, 24]])

print(arr)

print(type(arr))

arr[0, :] = series.values

print(arr)

arr2d = np.zeros((4, 6))

print(arr2d)

arr1D = np.array([1, 2, 8, 4, 5, 6, 7, 1, 9])

mask = arr1D >= 5

print(mask)

print(arr1D[2:4])

arr1DsubIndices = np.where(arr1D > 5)

print(arr1DsubIndices)

print(arr1D[arr1DsubIndices])

color = [
    f"rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})"
    for _ in range(25)
]

# print(color)


C = npmat.repmat([1, 2, 3], 5, 1)

print(C)


arrAssem = np.array([arr1D[0], arr1D[1]])

print(arrAssem)

date = pd.to_datetime("2019-03-21 9:00:00")

hours = 12  # hours
stepMin = 5  # minutes
nEvaluation = hours * 60 / stepMin

times = pd.date_range(
    date, date + pd.Timedelta(str(hours) + "h"), freq=str(stepMin) + "min"
)

x = arr1D[0 : len(arr1D) : 3]

print(x)


a = np.zeros(12, dtype=bool)

print(a)

b = np.zeros([10, 3])

b[3, :] = [1, 1, 1]
b[4] = [2, 2, 2]

print(b)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

b = np.ones(9, dtype=bool)

c = np.array([3, 1, 6])

for i in c:
    b[i] = False

d = a[b]

print(d)

aa = a[a == 5]

print(aa)

tf = np.array([True, False, False, True])

arr = np.array([1.3, 4.3, 6.5, 3.4])

# arr = np.append(arr, [9.4])

print(arr[tf])

startDate = str(2019) + "-01-01 12:30:00"
endDate = str(2019) + "-01-02 11:00:00"
times1 = pd.date_range(start=startDate, end=endDate, freq="H")
print(times1.timetz)

tz = "Europe/Stockholm"

times2 = pd.date_range(start=startDate, end=endDate, freq="H", tz=tz)
print(times2.timetz)

print(times1)
print(times2)

print(times1.hour)
print(times2.hour)


arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
sun = 5 * np.array([9, 2, 2])
print(arr + sun)

v1 = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3]])
v2 = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3]])
v3 = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3]])

fMid = (v1 + v2 + v3) / 3.0

# print(fMid)

arr = np.array([1.3, 4.3, 6.5, 3.4, 7.5, 2.4, 2.5, 1.4])

print(arr[0:3])


a = np.ones(1)
b = np.zeros(1)
c = np.ones(1)

mat = np.c_[a, b, c]

print(a)

face_data_dict = dict.fromkeys([fi for fi in range(0, 10)])

for i in range(0, 10):
    face_data_dict[i] = [0, 3, 4]

print(face_data_dict.items())

arr = np.array([1.3, 4.3, 6.5, 3.4, 7.5, 2.4, 2.5, 1.4])

# arrb = (arr > 2.4)

# print(int(arrb))
n = 10
max = 2 * np.pi
print(max)

ray_pts = np.zeros((10, 3))
print(ray_pts)

arr = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3], [0, 0, 0]])
print(arr)

pt = np.array([5, 0, 0])

pts = pt + arr

print(pts)

arr2D = np.array([[[1, 2, 3]], [[1, 5, 3]]])
print(arr)

arr3D = np.array(
    [
        [[1, 2, 3], [1, 5, 3], [1, 2, 3], [0, 0, 0]],
        [[11, 22, 13], [31, 25, 13], [21, 32, 43], [50, 60, 30]],
    ]
)
print(arr3D)

arrAdd = arr2D + arr3D

print(arrAdd)

arr3Dz = np.zeros([1, 2, 3])

arr3Dz[0, 0, :] = [6, 6, 6]

print(arr3Dz)


arr2Rep = np.array([[1, 2, 3]])

arrRep = np.repeat(arr2Rep, 5, axis=0)

print(arrRep)

print("------")
arr1 = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3], [5, 4, 3]])
arr2 = np.array([[3, 1, 2], [4, 3, 2], [6, 2, 2], [1, 2, 3]])
print(arr1)
print(arr2)

arr3 = np.cross(arr1, arr2)

arr3Len = np.sqrt((arr3**2).sum(-1))[..., np.newaxis]

indices = [0, 2]

print(arr3Len[indices])


arr1 = np.array([1, 5, 6])
arr2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(arr1)
print(arr2)

arr3 = np.delete(arr2, arr1)

print(arr3)

face_in_sun = np.array([])

print(face_in_sun)


ray_pts = np.zeros(6)

print(ray_pts)


arr1 = np.array([[1, 2, 3], [1, 5, 3], [1, 2, 3], [5, 4, 3]])
arr2 = np.array(["sdvsdv", "sdvsdv", "sdvbsdfbvs", "sdvsdvsdbfd"])
mask = [True, True, False, True]
print(arr1)
print(arr2)
arr1 = arr1[mask]
arr2 = arr2[mask]
print(arr1)
print(arr2)


start_date = "2015-03-30 01:00:00"
end_date = "2015-03-31 00:00:00"

sub_dates = pd.date_range(start=start_date, end=end_date, freq="1H")

print(sub_dates)


arr2 = np.array([1, 9, 3, 4, 2, 6, 7, 1, 9, 0, 9, 3])


dict_key = "2018-03-23T12:00:00"

print(len(dict_key))

year = dict_key[0:4]
sep1 = dict_key[4]
month = dict_key[5:7]
sep2 = dict_key[7]
day = dict_key[8:10]
sep3 = dict_key[10]
hour = dict_key[11:13]
sep4 = dict_key[13]
min = dict_key[14:16]
sep5 = dict_key[16]
sec = dict_key[17:19]

numbers = np.array([int(year), int(month), int(day), int(hour), int(min), int(sec)])
separators = np.array(
    [dict_key[4], dict_key[7], dict_key[10], dict_key[13], dict_key[16]]
)
facit = np.array(["-", "-", "T", ":", ":"])

print(numbers)

res_nan = np.all(np.isnan(numbers))  # np.isnan(numbers)
res_sep = np.array_equal(separators, facit)

print(res_sep)
print(res_nan)
print(year)
print(month)
print(day)
print(hour)
print(min)
print(sec)
print(separators)

dict_keys = [
    "2018-03-23T00:00:00",
    "2018-03-23T01:00:00",
    "2018-03-23T02:00:00",
    "2018-03-23T03:00:00",
    "2018-03-23T04:00:00",
    "2018-03-23T05:00:00",
    "2018-03-23T06:00:00",
    "2018-03-23T07:00:00",
    "2018-03-23T08:00:00",
    "2018-03-23T09:00:00",
    "2018-03-23T10:00:00",
    "2018-03-23T11:00:00",
    "2018-03-23T12:00:00",
    "2018-03-23T13:00:00",
    "2018-03-23T14:00:00",
    "2018-03-23T15:00:00",
    "2018-03-23T16:00:00",
    "2018-03-23T17:00:00",
    "2018-03-23T18:00:00",
    "2018-03-23T19:00:00",
    "2018-03-23T20:00:00",
    "2018-03-23T21:00:00",
    "2018-03-23T22:00:00",
    "2018-03-23T23:00:00",
]

time_from = "00:00:00"
time_to = "23:00:00"

hour_from = int(time_from[0:2])
hour_to = int(time_to[0:2])
# Remove the first and last
dict_keys_subset = dict_keys[hour_from : (len(dict_keys) - (24 - hour_to) + 1)]

pp(dict_keys_subset)


a = np.repeat("somefakekey", 5)

print(a.shape)

try:
    year = 2022
    month = 11
    day = 32
    t = pd.to_datetime(str(year) + "-" + str(month) + "-" + str(day))
except:
    day = 1
    month += 1
    if month > 12:
        month = 1

    t = pd.to_datetime(str(year) + "-" + str(month) + "-" + str(day))
    print(t)

date = pd.to_datetime("2020-1-1")
print(date)

a = np.array([[1, 2, 3], [4, 2, 1], [2, 3, 1]])
b = np.array([6, 8, 3], dtype=float)

dist = np.sqrt(np.sum((a - b) ** 2, axis=1))

avrg = np.average(a)


a = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])

x = a[0::6]
y = a[1::6]
z = a[2::6]

xyz = a[0:3:6]

print(x)
print(y)
print(z)

print(xyz)


def _calc_n_sides(n_points: int):
    low_count = 1000000
    upper_count = 15000000

    low_sides = 5
    upper_sides = 12

    count_diff = upper_count - low_count
    sides_diff = upper_sides - low_sides

    if n_points < low_count:
        n_sides = upper_sides
    elif n_points > upper_count:
        n_sides = low_sides
    else:
        n_sides = low_sides + sides_diff * (1 - ((n_points - low_count) / count_diff))
        n_sides = round(n_sides, 0)

    return n_sides


n_sides = _calc_n_sides(16000000)

print(n_sides)

c = np.array([[0, 0, 0]])

a = np.array([[1, 2, 3], [3, 2, 1]])
b = np.array([[6, 5, 4], [4, 5, 6]])

c = np.concatenate((c, a), axis=0)
c = np.concatenate((c, b), axis=0)

print(c)

c = np.delete(c, obj=0, axis=0)

print(c)

print("testing")

a = np.array([1, 2, 3, 4, 3, 3, 2, 3, 4, 5, 3, 43, 54, 3, 3, 4])

a = np.array([[1, 2, 3], [4, 3, 3], [2, 3, 4], [5, 3, 43], [54, 3, 3]])

pp(a)

print(a[:, 0])

b = np.where(a[:, 0] > 2)

print(b)

print(a[b])


a = 150000
b = 300000

c = range(a, b, 12)


print(c)

a = np.array()
b = np.array([[1, 2, 3], [4, 3, 3], [2, 3, 4], [5, 3, 43], [54, 3, 3]])
c = np.array([[1, 2, 3], [4, 3, 3], [2, 3, 4], [5, 3, 43], [54, 3, 3]])

d = np.concatenate((a, b))

e = np.vstack([a, b])

print(e)

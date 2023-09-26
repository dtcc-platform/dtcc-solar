from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.system("clear")

tz = "Asia/Calcutta"
lat, lon = 28.6, 77.2

times = pd.date_range("2019-01-01 00:00:00", "2020-01-01", freq="H", tz=tz)

print(times.values)

solpos = solarposition.get_solarposition(times, lat, lon)

# remove nighttime
solpos = solpos.loc[solpos["apparent_elevation"] > 0, :]

ax = plt.subplot(1, 1, 1, projection="polar")

# draw the analemma loops
points = ax.scatter(
    np.radians(solpos.azimuth),
    solpos.apparent_zenith,
    s=2,
    label=None,
    c=solpos.index.dayofyear,
)
ax.figure.colorbar(points)

# draw hour labels
for hour in np.unique(solpos.index.hour):
    # choose label position by the smallest radius for each hour
    subset = solpos.loc[solpos.index.hour == hour, :]
    r = subset.apparent_zenith
    pos = solpos.loc[r.idxmin(), :]
    ax.text(np.radians(pos["azimuth"]), pos["apparent_zenith"], str(hour))

# draw individual days
for date in pd.to_datetime(["2019-03-21", "2019-06-21", "2019-12-21"]):
    times = pd.date_range(date, date + pd.Timedelta("24h"), freq="5min", tz=tz)
    solpos = solarposition.get_solarposition(times, lat, lon)

    solpos = solpos.loc[solpos["apparent_elevation"] > 0, :]
    label = date.strftime("%Y-%m-%d")
    ax.plot(np.radians(solpos.azimuth), solpos.apparent_zenith, label=label)

# print(solpos)

# utils.ExportResults(solpos)


ax.figure.legend(loc="upper left")

# change coordinates to be like a compass
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rmax(90)

plt.show()

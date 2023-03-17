from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import math
import numpy.matlib as npmat

os.system('clear')

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,1,1,1,1,1,1,1,1,1]
z = [1,1,1,1,1,1,1,1,1,1]
colors = [1,2,3,4,5,6,7,8,9,10]

ax = plt.axes(projection = '3d')
ax.scatter3D(x,y,z,c=colors,s=100,cmap='Reds', vmin = 0, vmax = 5)


plt.show()


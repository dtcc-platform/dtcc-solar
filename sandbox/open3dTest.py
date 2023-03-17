
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d

os.system('clear')

armadillo = open3d.data.ArmadilloMesh()

mesh = open3d.io.read_triangle_mesh(armadillo.path)

print(mesh)




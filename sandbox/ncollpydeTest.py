import numpy as np
import os

os.system('clear')

# get an array of vertices and triangles which refer to those points
import meshio
mesh = meshio.read("teapot.stl")

# use this library
from ncollpyde import Volume

volume = Volume(mesh.points, mesh.cells_dict["triangle"])
#volume = Volume.from_meshio(mesh) # or, for convenience

avrgPt = [0,0,0]
nPts = len(mesh.points)
for p in mesh.points:
    avrgPt[0] += p[0] / nPts
    avrgPt[1] += p[1] / nPts
    avrgPt[2] += p[2] / nPts

print(avrgPt)


# containment checks: singular and multiple
assert [-2.30, -4.15,  1.90] in volume
assert np.array_equal(
    volume.contains(
        [
            [-2.30, -4.15, 1.90],
            [-0.35, -0.51, 7.61],
        ]
    ),
    [True, False]
)

# line segment intersection, lines are created between source points and target points.
#seg_idxs, intersections, is_backface = volume.intersections(
#    [[-10, -10, -10], [0, 0, 3], [-20, 0, 5]],
#    [[0, 0, 3], [10, 10, 10], [20, 0, 5]],
#)

ptOrigin = []
ptTarget = []

for i in range(0,5):
    ptOrigin.append(avrgPt)
    rnd = 1000 * np.random.rand(1,3)
    if i == 2:
        rnd =  0.01 * np.random.rand(1,3)

    ptTarget.append(rnd[0].tolist())

print(ptOrigin)
print(ptTarget)    

seg_idxs, intersections, is_backface = volume.intersections(ptOrigin,ptTarget,)

print("---- Intersection results ----")
print(seg_idxs)
print(intersections)
print(is_backface)




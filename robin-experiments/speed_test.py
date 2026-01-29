import plyfile
import numpy as np
import time
from combine_splats import transform_splat_data

ply = plyfile.PlyData.read(
    r"D:\splatter-node\worker\data\jobs\dd25ab185e71\input\test.ply"
)

vert = ply["vertex"].data

scale = 0.99
R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
t = np.array([-0.002, 0.001, 0.01])

start_time=time.time()
vert2 = transform_splat_data(vert, scale, R, t)
print("TIME: ", time.time() - start_time)
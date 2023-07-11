import numpy as np
from typing import Tuple, Union
from math import sin, cos, atan2, sqrt

inputs = np.load('dataset/weather_time_gustsfc_cut/spherical_points.npy')
print(inputs.shape)

max_theta = np.amax(inputs[:,0])
max_phi = np.amax(inputs[:,1])
min_theta = np.amin(inputs[:,0])
min_phi = np.amin(inputs[:,1])
print('min, max theta : ',min_theta, max_theta)
print('min, max phi : ',min_phi, max_phi)

# for point in inputs:
#     theta = point[0]
#     phi = point[1]
    # if theta<0 or phi<0:
    #     print('theta, phi : ',theta, phi)
    #     print("There is minus")
    #     exit(0)



# Number = Union[int, float]
# Vector = Tuple[Number, Number, Number]


# def distance(a: Vector, b: Vector) -> Number:
#     """Returns the distance between two cartesian points."""
#     x = (b[0] - a[0]) ** 2
#     y = (b[1] - a[1]) ** 2
#     z = (b[2] - a[2]) ** 2
#     return (x + y + z) ** 0.5

  
# def magnitude(x: Number, y: Number, z: Number) -> Number:
#     """Returns the magnitude of the vector."""
#     return sqrt(x * x + y * y + z * z)


# def to_spherical(x: Number, y: Number, z: Number) -> Vector:
#     """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
#     radius = magnitude(x, y, z)
#     theta = atan2(sqrt(x * x + y * y), z)
#     phi = atan2(y, x)
#     return (theta, phi)

# def to_spherical(x, y, z):
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arctan2(y, x)  # range of atan2 is [-pi, pi]
#     theta = theta if theta >= 0 else theta + 2*np.pi  # shift range to [0, 2pi]
#     phi = np.arccos(z / r)  # range of arccos is [0, pi]
#     return theta, phi


# target = 'dpt2m'# 'tcdcclm' # 'gustsfc'
# targets = ['dpt2m', 'tcdcclm' , 'gustsfc']

# for target in targets : 
#     points = np.load('./dataset/weather_time_'+target+'_cut/points.npy')
#     # print('points shape : ',points.shape)
#     # print(points[0])
#     spherical_points = []
#     for point in points:
#         spherical_points.append(to_spherical(point[0], point[1], point[2]))

#     np_spherical_points = np.array(spherical_points)
#     # print('np_spherical_points shape : ',np_spherical_points.shape)
#     # print(np_spherical_points[0:10])

#     np.save("./dataset/weather_time_"+target+"_cut/spherical_points.npy",np.array(spherical_points) )
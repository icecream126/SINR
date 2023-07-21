import numpy as np

def to_spherical(x, y, z):
    '''
    xyz to spherical coordinates
    theta : [0, \pi]
    phi : [0, 2\pi]
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)  # range of arctan2 is [-pi, pi]
    phi = phi if phi >= 0 else phi + 2*np.pi  # shift range to [0, 2pi]
    theta = np.arccos(z / r)  # range of arccos is [0, pi]
    return theta, phi

targets = ['dpt2m', 'tcdcclm' , 'gustsfc']

for target in targets : 
    points = np.load('./dataset/weather_time_'+target+'_cut/points.npy')
    spherical_points = []
    for point in points:
        spherical_points.append(to_spherical(point[0], point[1], point[2]))

    np_spherical_points = np.array(spherical_points)
    np.save("./dataset/weather_time_"+target+"_cut/spherical_points.npy",np.array(spherical_points) )
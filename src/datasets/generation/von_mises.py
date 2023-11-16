import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv
import os

'''
# Single mu
def vmf_pdf(x, mu, kappa):
    d = len(mu)
    C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
    return C_d * np.exp(kappa * mu.dot(x))


def main(args):
    data  = {}
    kappa = args.kappa
    mu = np.array(args.mu).reshape(-1, 3)
    res = args.res
    # latitude_res, longitude_res = args.resolution
    
    # latitude, longitude = np.mgrid[-np.pi/2:np.pi/2:latitude_res*1j, 0.0:2.0*np.pi:longitude_res*1j]
    latitude = np.linspace(-90, 90, int(180/res))
    longitude = np.linspace(0, 360, int(360/res))
    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)
    
    # print(longitude.min(), longitude.max())
    # print(latitude.min(), latitude.max())
    
    data['latitude'] = latitude
    data['longitude'] = longitude
    
    latitude, longitude = np.meshgrid(latitude, longitude, indexing='ij')
    
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # x_min, y_min, z_min = np.inf, np.inf, np.inf
    # x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    
    # for kappa, mu in zip(kappas, mus):
    # mu = args.mu
    # kappa = args.kappa
    pdf_values = np.zeros_like(z)
    for i in range(pdf_values.shape[0]):
        for j in range(pdf_values.shape[1]):
            pdf_values[i, j] = vmf_pdf(np.array([x[i, j], y[i, j], z[i, j]]), mu, kappa)

    data['target'] = pdf_values
        
    # pdf_values /= pdf_values.max()
    # x_adj = x * (1 + pdf_values)
    # y_adj = y * (1 + pdf_values)
    # z_adj = z * (1 + pdf_values)
    
    # x_min = min(x_min, x_adj.min())
    # y_min = min(y_min, y_adj.min())
    # z_min = min(z_min, z_adj.min())
    # x_max = max(x_max, x_adj.max())
    # y_max = max(y_max, y_adj.max())
    # z_max = max(z_max, z_adj.max())

    # ax.plot_surface(x_adj, y_adj, z_adj, rstride=1, cstride=1, facecolors=plt.cm.coolwarm(pdf_values), alpha=0.6, antialiased=True)
    
    # absolute_max = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), abs(z_min), abs(z_max))
    # ax.set_xlim([-absolute_max, absolute_max])
    # ax.set_ylim([-absolute_max, absolute_max])
    # ax.set_zlim([-absolute_max, absolute_max])
    # ax.set_box_aspect([absolute_max] * 3)
    
    # plt.title('von Mises-Fisher distribution')
    # plt.savefig('./von_mises2.png')
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f"{args.output_dir}/{args.custom_fn}", **data)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Multiple von Mises-Fisher Distributions on Sphere.')
    parser.add_argument('--kappa', type=float, default=80.0, help='List of concentration parameters for the von Mises-Fisher distributions.')
    parser.add_argument('--mu', type=float, nargs='+', default=[0.0, 0.0, 1.0], help='List of mean directions for the von Mises-Fisher distributions. Should be multiple of 3.')
    parser.add_argument('--res', type=int, default=1, help='Resolution of latitude and longitude.')
    parser.add_argument('--output_dir', type=str, default='dataset/spatial/von_mises')
    parser.add_argument('--custom_fn', type=str, default='data')
    args = parser.parse_args()
    args.output_dir = args.output_dir+'_'+str(int(args.kappa))+'_'+ str(int(args.mu[0]))+ str(int(args.mu[1]))+ str(int(args.mu[2]))
    main(args)
    
'''
'''
# Multiple mus
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv


def vmf_pdf(x, mu, kappa):
    d = len(mu)
    C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
    return C_d * np.exp(kappa * mu.dot(x))


def main(args):
    kappas = args.kappa
    mus = np.array(args.mu).reshape(-1, 3)
    latitude_res, longitude_res = args.resolution
    
    latitude, longitude = np.mgrid[-np.pi/2:np.pi/2:latitude_res*1j, 0.0:2.0*np.pi:longitude_res*1j]
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    
    for kappa, mu in zip(kappas, mus):
        pdf_values = np.zeros_like(z)
        for i in range(pdf_values.shape[0]):
            for j in range(pdf_values.shape[1]):
                pdf_values[i, j] = vmf_pdf(np.array([x[i, j], y[i, j], z[i, j]]), mu, kappa)
        
        pdf_values /= pdf_values.max()
        x_adj = x * (1 + pdf_values)
        y_adj = y * (1 + pdf_values)
        z_adj = z * (1 + pdf_values)
        
        x_min = min(x_min, x_adj.min())
        y_min = min(y_min, y_adj.min())
        z_min = min(z_min, z_adj.min())
        x_max = max(x_max, x_adj.max())
        y_max = max(y_max, y_adj.max())
        z_max = max(z_max, z_adj.max())

        ax.plot_surface(x_adj, y_adj, z_adj, rstride=1, cstride=1, facecolors=plt.cm.viridis(pdf_values), alpha=0.6, antialiased=True)
    
    absolute_max = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), abs(z_min), abs(z_max))
    ax.set_xlim([-absolute_max, absolute_max])
    ax.set_ylim([-absolute_max, absolute_max])
    ax.set_zlim([-absolute_max, absolute_max])
    ax.set_box_aspect([absolute_max] * 3)
    
    plt.title('Multiple von Mises-Fisher Distributions on Sphere with Latitude and Longitude')
    plt.savefig('./multiple_von_mises.png')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Multiple von Mises-Fisher Distributions on Sphere.')
    parser.add_argument('--kappa', type=float, nargs='+', default=[20.0, 50.0], help='List of concentration parameters for the von Mises-Fisher distributions.')
    parser.add_argument('--mu', type=float, nargs='+', default=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], help='List of mean directions for the von Mises-Fisher distributions. Should be multiple of 3.')
    parser.add_argument('--resolution', type=int, nargs=2, default=[100, 100], help='Resolution of latitude and longitude.')
    
    args = parser.parse_args()
    main(args)
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import iv
import random


def vmf_pdf(x, mu, kappa):
    d = len(mu)
    C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
    return C_d * np.exp(kappa * mu.dot(x))


def generate_random_mu(num_points=5):
    mu_values = []
    for _ in range(num_points):
        z = random.uniform(-1, 1)  # Sample z uniformly from [-1, 1]
        phi = np.arccos(z)  # Compute phi
        
        theta = random.uniform(0, 2 * np.pi)  # Sample theta uniformly from [0, 2*pi]
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        mu_values.append([x, y, z])
    
    return np.array(mu_values)


def main(args):
    data = {}
    kappas = args.kappa
    mus = generate_random_mu(args.num_points)  # Here, mus are generated randomly.
    # latitude_res, longitude_res = args.resolution
    
    # latitude, longitude = np.mgrid[-np.pi/2:np.pi/2:latitude_res*1j, 0.0:2.0*np.pi:longitude_res*1j]
    
    lat_deg = np.linspace(-90, 90, int(180/args.res))  # Latitude: -90 to 90 degrees
    lon_deg = np.linspace(0, 360, int(360/args.res))  # Longitude: 0 to 360 degrees
    lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    
    data['latitude']=lat_rad
    data['longitude']=lon_rad
    
    
    latitude, longitude = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
        
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    
    all_pdf_values=[]
    
    for kappa, mu in zip(kappas, mus):
        pdf_values = np.zeros_like(z)
        for i in range(pdf_values.shape[0]):
            for j in range(pdf_values.shape[1]):
                pdf_values[i, j] = vmf_pdf(np.array([x[i, j], y[i, j], z[i, j]]), mu, kappa)
        
        pdf_values /= pdf_values.max()
        all_pdf_values.append(pdf_values)
        
        x_adj = x * (1 + pdf_values)
        y_adj = y * (1 + pdf_values)
        z_adj = z * (1 + pdf_values)
        
        x_min = min(x_min, x_adj.min())
        y_min = min(y_min, y_adj.min())
        z_min = min(z_min, z_adj.min())
        x_max = max(x_max, x_adj.max())
        y_max = max(y_max, y_adj.max())
        z_max = max(z_max, z_adj.max())

        ax.plot_surface(x_adj, y_adj, z_adj, rstride=1, cstride=1, facecolors=plt.cm.viridis(pdf_values), alpha=0.6, antialiased=True)
    
    data['target']=all_pdf_values
    # import pdb
    
    # pdb.set_trace()
    
    print('data[latitude] : ',data['latitude'].shape)
    # print('data[longitude] : ',data['longitude'].shape)
    # print('data[target] : ',data['target'].shape)
    
    absolute_max = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), abs(z_min), abs(z_max))
    ax.set_xlim([-absolute_max, absolute_max])
    ax.set_ylim([-absolute_max, absolute_max])
    ax.set_zlim([-absolute_max, absolute_max])
    ax.set_box_aspect([absolute_max] * 3)
    
    plt.title('Multiple von Mises-Fisher Distributions on Sphere with Latitude and Longitude')
    plt.savefig('./final_multiple_von_mises.png')
    # plt.show()
    np.savez(f'{args.output_dir}/{args.custom_fn}',**data)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Multiple von Mises-Fisher Distributions on Sphere.')
    parser.add_argument('--kappa', type=float, nargs='+', default=[40.0, 50.0, 60.0, 70.0, 80.0], help='List of concentration parameters for the von Mises-Fisher distributions.')
    parser.add_argument('--num_points', type=int, default=5, help='Number of random mu points to generate.')
    parser.add_argument('--res', type=int, default=1, help='Resolution of latitude and longitude.')
    parser.add_argument('--custom_fn', type=str, default='data.npz', help='Filename to save the data and image.')
    parser.add_argument('--output_dir', type=str, default='dataset/spatial/von_mises', help='Directory to save the data and image.')

    args = parser.parse_args()
    args.output_dir = args.output_dir+'_'+str(args.kappa)+'_'+str(args.num_points)
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

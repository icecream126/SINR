# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def generate_line_wave_data(num_points=1000, k=5):
#     # Generate random points on the sphere
#     theta = 2 * np.pi * torch.rand(num_points)  # Longitude
#     phi = torch.acos(2 * torch.rand(num_points) - 1)  # Latitude

#     x = torch.sin(phi) * torch.cos(theta)
#     y = torch.sin(phi) * torch.sin(theta)
#     z = torch.cos(phi)

#     # Compute the wave values using the sine function along the longitude
#     values = torch.sin(k * theta)

#     return torch.stack([x, y, z, values], dim=1)

# def visualize_spherical_data(data):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     x, y, z, values = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
#     sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=20)
#     plt.colorbar(sc)
#     ax.set_box_aspect([1, 1, 1])
#     plt.savefig('./myfigure.png')
#     plt.show()

# # Generate and visualize the dataset
# line_wave_data = generate_line_wave_data()
# visualize_spherical_data(line_wave_data)

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def generate_spherical_data(num_points, k):
    # Generate random points on a sphere using spherical coordinates
    theta = 2 * np.pi * torch.rand(num_points)  # Longitude
    phi = torch.acos(2 * torch.rand(num_points) - 1)  # Latitude

    # Compute the Cartesian coordinates of the points
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Generate wave values based on the longitude
    values = torch.sin(k * phi)

    return torch.stack([x, y, z], dim=1), values


def visualize_spherical_data(points, values, k):
    # Create a mesh grid for visualization
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Use RBF interpolation to get values for the mesh grid
    rbf = Rbf(
        points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), values.numpy()
    )
    interpolated_values = rbf(X, Y, Z)

    # Plot the sphere with the interpolated values
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.jet(interpolated_values),
        alpha=0.6,
    )
    ax.set_title(f"Spherical Data with {k} Oscillations")
    plt.savefig("./myfigure_phi.png")
    plt.show()


# Generate and visualize the data
num_points = 1000
k = 5  # Number of oscillations
points, values = generate_spherical_data(num_points, k)
visualize_spherical_data(points, values, k)

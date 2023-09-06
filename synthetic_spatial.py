import torch
import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt


def generate_spherical_oscillation_data(num_points=1000, k=5):
    # Generate random points on a sphere using spherical coordinates
    theta = 2 * np.pi * torch.rand(num_points)  # Longitude
    phi = torch.acos(2 * torch.rand(num_points) - 1)  # Latitude

    # Compute the Cartesian coordinates of the points
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Generate wave values based on a combination of longitude and latitude for spatial oscillation
    values = torch.sin(k * theta + phi)

    return torch.stack([x, y, z], dim=1), values


def visualize_spherical_oscillation(points, values):
    # Create a mesh grid for visualization on the sphere
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
        facecolors=plt.cm.viridis(interpolated_values),
        alpha=0.6,
    )
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Continuous Spatial Oscillation on Sphere")
    plt.savefig("./continuous_spherical_oscillation.png")
    plt.show()


# Visualize the continuous oscillation on the sphere
spherical_oscillation_data, values = generate_spherical_oscillation_data()
visualize_spherical_oscillation(spherical_oscillation_data, values)

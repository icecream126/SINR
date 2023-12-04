import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import os


# Define the Kent distribution's PDF
def kent_pdf(x, mu, kappa, beta, gamma1, gamma2):
    C = 1  # For simplicity, you may need to calculate the normalization constant C depending on your parameters
    return C * np.exp(kappa * mu.dot(x) + beta * (gamma1.dot(x)) * (gamma2.dot(x)))


def visualize_and_save(data, filename):
    lat_rad = data["latitude"]
    lon_rad = data["longitude"]
    pdf_values = data["target"]

    latitude, longitude = np.meshgrid(lat_rad, lon_rad, indexing="ij")
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    pdf_values /= pdf_values.max()
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.viridis(pdf_values),
        alpha=0.6,
        antialiased=True,
    )
    plt.title("Kent Distribution on Sphere with Latitude and Longitude")
    plt.savefig(filename + ".png")
    # plt.show()
    np.savez(filename, **data)


def main():
    kappa_values = [30.0, 40.0, 50.0]  # Example kappa values
    beta_values = [5.0, 10.0, 15.0]  # Example beta values
    mu_values = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Example mu values
    gamma1_values = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Example gamma1 values
    gamma2_values = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Example gamma2 values

    lat_deg = np.linspace(-90, 90, 180)
    lon_deg = np.linspace(0, 360, 360)
    lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)

    output_dir = "kent_distribution_datasets"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(3):  # Generate three point sets sampled from the Kent distribution
        data = {"latitude": lat_rad, "longitude": lon_rad}
        latitude, longitude = np.meshgrid(lat_rad, lon_rad, indexing="ij")
        x = np.cos(latitude) * np.cos(longitude)
        y = np.cos(latitude) * np.sin(longitude)
        z = np.sin(latitude)

        mu = mu_values[i]
        kappa = kappa_values[i]
        beta = beta_values[i]
        gamma1 = gamma1_values[i]
        gamma2 = gamma2_values[i]

        pdf_values = np.zeros_like(z)
        for lat_idx in range(len(lat_rad)):
            for lon_idx in range(len(lon_rad)):
                pdf_values[lat_idx, lon_idx] = kent_pdf(
                    np.array(
                        [x[lat_idx, lon_idx], y[lat_idx, lon_idx], z[lat_idx, lon_idx]]
                    ),
                    mu,
                    kappa,
                    beta,
                    gamma1,
                    gamma2,
                )

        data["target"] = pdf_values
        filename = os.path.join(output_dir, f"kent_distribution_{i}")
        visualize_and_save(data, filename)


if __name__ == "__main__":
    main()

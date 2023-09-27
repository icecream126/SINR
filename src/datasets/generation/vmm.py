import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.special import iv

# Define von Mises-Fisher PDF
def vmf_pdf(x, mu, kappa):
    d = len(mu)
    C_d = kappa ** (d / 2 - 1) / ((2 * np.pi) ** (d / 2) * iv(d / 2 - 1, kappa))
    return C_d * np.exp(kappa * mu.dot(x))

def visualize_and_save(data, filename):
    lat_rad = data['latitude']
    lon_rad = data['longitude']
    pdf_values = data['target']
    
    latitude, longitude = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    pdf_values /= pdf_values.max()
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(pdf_values), alpha=0.6, antialiased=True)
    plt.title('Mixture of von Mises-Fisher Distributions on Sphere with Latitude and Longitude')
    plt.savefig(filename + '.png')
    # plt.show()
    np.savez(filename, **data)

def main():
    # Example weights, mu's and kappa's
    weights = [0.3, 0.4, 0.3]
    mu_values = np.array([[0, 0, 1], [1, 0, 0], [np.sqrt(2)/2.0, -np.sqrt(2)/2.0, 0]])
    kappa_values = [20.0, 50.0, 80.0]

    lat_deg = np.linspace(-90, 90, 180)  
    lon_deg = np.linspace(0, 360, 360) 
    lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)

    output_dir = './dataset/spatial/vmm/'+str(int(kappa_values[0]))+'_'+str(int(kappa_values[1]))+'_'+str(int(kappa_values[2]))
    os.makedirs(output_dir, exist_ok=True)

    data = {'latitude': lat_rad, 'longitude': lon_rad}
    latitude, longitude = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    mixture_pdf_values = np.zeros_like(z)
    for weight, mu, kappa in zip(weights, mu_values, kappa_values):
        pdf_values = np.zeros_like(z)
        for lat_idx in range(len(lat_rad)):
            for lon_idx in range(len(lon_rad)):
                pdf_values[lat_idx, lon_idx] = vmf_pdf(np.array([x[lat_idx, lon_idx], y[lat_idx, lon_idx], z[lat_idx, lon_idx]]), mu, kappa)
        
        mixture_pdf_values += weight * pdf_values  # forming the mixture pdf
    
    data['target'] = mixture_pdf_values
    filename = os.path.join(output_dir, 'data')
    visualize_and_save(data, filename)

if __name__ == "__main__":
    main()
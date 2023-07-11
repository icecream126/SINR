import plotly.graph_objects as go

from scipy.special import sph_harm

import numpy as np

# Define the 1-dimensional coordinate axes:
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)

# Make a 2D-meshed grid out of the axes:
Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

def create_figure(l, m):
    """Create a figure of Y_lm using Plotly."""

    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2*np.pi, 100)
    
    Thetas, Phis = np.meshgrid(thetas, phis)
    
    ylm = sph_harm(m, l, Phi, Theta)

    fcolors = ylm.real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)

    R = abs(ylm)
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    
    fig = go.Figure(
                data=[
                    go.Surface(
                        x=X, y=Y, z=Z, 
                        surfacecolor=fcolors,
                        colorscale='balance', showscale=False, 
                        opacity=1.0, hoverinfo='none',
                    )
                ]
    )

    fig.update_layout(title='$Y_{%d%d}$' % (l, m), autosize=False,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90)
    )
    fig.show()
    
    return fig

create_figure(1, 2)

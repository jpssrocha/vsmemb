"""
Functions to help visualizing results.


Code from Jake VPD blog: 

https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, edgecolor="k", lw=2, **kwargs))
        

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=2, cmap='viridis', zorder=2, alpha=0.2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=2, zorder=2, alpha=0.2)
    ax.axis('equal')
    
    w_factor = 0.1 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

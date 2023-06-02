"""
Module containing functions to help calculate initial kicks for the 2 Gaussian
components on the VS method.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from astropy.coordinates import SkyCoord
import astropy.units as u


def _angular_size(physical_size: float, distance: float) -> float:
    """
    Given the physical size (`physical_size`) and `distance` calculate the
    angular size `alpha` from that distance.

    OBS: Inputs must be in the same unit.

    Parameters
    ----------
        physical_size: float
            actual size of the object

        distance: float
            distance to that object

    Returns
    -------
        alpha: float
            angular size in radians
    """
    
    theta: float = np.arctan( physical_size/(2*distance) )
    alpha: float  = 2*theta

    return alpha


def get_mean_initial_guess(
    cat: np.ndarray,
    std_filter: None | float = None,
    plot: bool = False
    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    From ndarray of proper motions (ra, dec) get initial guesses for the mean of the
    two gaussians on the mixture by looking the the two most proeminent peaks
    on the 2D histogram.
    
    Parameters:
    -----------
        cat: ndarray
            ndarray of proper motions
            
        std_filter: float or None. default = None
            If not none apply a gaussian blur with the value of std_filter.
            
        plot: bool. default: False
            if true plot the 2D histogram with the initial guesses.
    
    Returns
    -------
        : tuple, tuple
            A tuple for the x coordinates and one for the y coordinates
    """
    
    
    im, xe, ye = np.histogram2d(cat[:, 0], cat[:, 1], bins=50)
    
    im = im.T  # The histograms from np.histogram2d are kind of weird ...
               # i'm transposing it to return to the conventional numpy intuition (y, x)
    
    if std_filter:
        im = gaussian_filter(im, std_filter)
    
    coordinates = peak_local_max(im, num_peaks=2)
    
    y1, x1 = coordinates[0]
    y2, x2 = coordinates[1]
    
    y1, x1 = ye[y1], xe[x1]
    y2, x2 = ye[y2], xe[x2]
    
    if plot:
        plt.imshow(im[::-1], extent=[xe[0], xe[-1], ye[0], ye[-1]])
        plt.scatter([x1, x2], [y1, y2], s=5)
        plt.xlim(xe.min(), xe.max())
        plt.ylim(ye.min(), ye.max())
        plt.show()
        plt.scatter(cat[:, 0], cat[:, 1], s=5, alpha=0.3)
        plt.scatter([x1, x2], [y1, y2], s=10)
        plt.show()       


    return (x1, x2), (y1, y2)


def get_initial_weights(
    cat: pd.DataFrame, 
    info: pd.Series,
    factor: float = 1.0,
    plot: bool = False,
    ) -> tuple[float, float]:
    """
    Given a Gaia Catalog of a cluster, it's informations given by Baumgardt 2023
    and a multiplicative factor to scale the core radius of the cluster to define
    a central region, calculate a initial kick for the weights on the Gaussian
    mixture.
    
    Parameters
    ----------
    
    cat: pd.DataFrame
        Gaia DR3 catalog
    
    info: pd.Series
        Baumeister line refering to the cluster
    
    factor: float, default: 1.0
        Multiplicative factor to define central region
    
    Returns
    -------
        w1, w2: float, float
            Weights for two Gaussians 
    """
    
    rc = _angular_size(info.rc, info.R_Sun)*180/np.pi
    center = SkyCoord(ra=info.RA*u.deg, dec=info.DEC*u.deg, frame="icrs")
    stars = SkyCoord(ra=cat.ra*u.deg, dec=cat.dec*u.deg, frame="icrs")
    
    sep = center.separation(stars).value
    mask = sep < factor*rc
    
    w1, w2 = mask.mean(), (~mask).mean()
    
    if plot:
        plt.scatter(cat.ra, cat.dec, s=5, c=(sep < factor*rc), alpha=0.5, cmap="viridis")
        plt.title(f"{factor*rc = }")
        plt.show()
    
    return w1, w2


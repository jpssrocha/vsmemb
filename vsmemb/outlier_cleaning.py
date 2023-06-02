"""
Module with functions to clear outliers and give back the array for the
fitting procedure.
"""
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clear_outliers(
        cat: pd.DataFrame,
        n_neighboors: int = 20,
        plot: bool = False
    )-> np.ndarray:
    """
    Given the DR3 catalog it removes outliers and gives the array ready to
    operate.
    
    Parameters
    ----------
    
    cat: DataFrame
        Gaia DR3 catalog.
    
    plot: bool, default: False
        If true show the outliers.
    
    Returns
    -------
    
    X2: ndarray
         array with the proper motions (pmra, pmdec) cleared of outliers.
    """
    
    X = cat.loc[:, ["pmra", "pmdec"]].values
    lof = LocalOutlierFactor(n_neighbors=n_neighboors, contamination="auto")
    good: np.ndarray = lof.fit_predict(X) == 1
    
    if plot:
        plt.scatter(X[:, 0], X[:, 1], s=5, label="Full")
        plt.scatter(X[good, 0], X[good, 1], s=1, label="Cleaned")
        plt.legend()
        plt.show()
        
    X2 = np.copy(X[good])

    return X2

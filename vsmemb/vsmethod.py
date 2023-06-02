"""
Main method from this package.
"""

# std lib
from copy import deepcopy

# 3rt party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# local
from .initial_kicks import get_initial_weights, get_mean_initial_guess
from .outlier_cleaning import clear_outliers
from .plots import plot_gmm


def get_probabilities(
    cat: pd.DataFrame,
    info: pd.Series,
    n_neighboors: int = 15,
    factor: float = 1,
    overlap_mean: bool = False,
    runs: int = 25,
    plots: bool = False,
    ) -> tuple[pd.DataFrame, list[float], GaussianMixture, int]:
    """
    Given a Gaia DR3 catalog for a cluster and it's information present in
    Baumgardt 2023.
    
    Parameters
    ----------
    
    cat: pd.DataFrame
        Gaia DR3 catalog for a cluster.
    
    info: pd.Series
        Line from Baumgardt with info about the cluster.

    n_neighboors: int, default: 15
        Number of neighbors to use while performing KNN based outlier cleaning.

    factor: float, default: 1.0
        Scalling factor for the cluster core radius to estimate weights.

    overlap_mean: bool, default: False
        If true use the same mean estimation for both clusters.

    runs: int, default: 25
        Number of runs for each combination of initial_kicks (total number of
        runs = runs*4.

    plots: bool, default: False
        If true plot the visualizations regarding the internal steps.

    
    Returns
    -------
    
    cat: pd.DataFrame
        Catalog added with the probability column
    """       
    # Clear outliers
    X = cat.loc[:, ["pmra", "pmdec"]].values
    X2 = clear_outliers(cat, n_neighboors=n_neighboors, plot=(True and plots))
    
    # Get inital kicks
    w1, w2 = get_initial_weights(cat, info, factor=factor, plot=(True and plots))
    xs, ys = get_mean_initial_guess(X2, plot=(True and plots))
    [x1_c, x2_c], [y1_c, y2_c] = xs, ys
    
    # permutations of initial kicks
    # ugly ... but i'm too tired to think of a better way to permutate ...
    M = [ [ [x1_c, y1_c], [x2_c, y2_c] ], [ [x2_c, y2_c], [x1_c, y1_c] ]  ]
    W = [ [w1, w2], [w2, w1] ]
    
    if overlap_mean:
         M = [ [ [x1_c, y1_c], [x1_c, y1_c] ], [ [x2_c, y2_c], [x2_c, y2_c] ]  ]
    
    combinations = []
    
    for weights in W: 
        for means in M:
            combinations.append([weights, means])
    
    # Run GMM
    scores = []
    models = []

    for comb in combinations:
        weights, means = comb
        for _ in range(runs):
            gmm2 = GaussianMixture(
                n_components=2,
                covariance_type='full',
                weights_init=weights,
                means_init=means,
                warm_start=False,
            )

            gmm2.fit(X2)
            models.append(deepcopy(gmm2))
            scores.append(gmm2.score(X))
            
    # Select best model
    ind_best: int = scores.index(max(scores))
    
    # Select cluster component
    var: list[float] = []
    for cov in models[ind_best].covariances_:
        var.append(np.diag(cov).sum())
        
    ind_cluster: int = var.index(min(var))
    probs: np.ndarray = models[ind_best].predict_proba(X)[:, ind_cluster]/(np.sum(models[ind_best].predict_proba(X), axis=1))
    
    # Put probability on the catalog
    cat["membership_prob"] = probs

    if plots:
        plt.scatter(x="pmra", y="pmdec", c="membership_prob", s=1, alpha=0.5, data=cat)
        plt.colorbar()
        plt.show()
        plot_gmm(models[ind_best], cat.loc[:, ["pmra", "pmdec"]].values)
        plt.show()
    
    return cat, scores, models[ind_best], ind_cluster

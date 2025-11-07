import numpy as np
from scipy.stats import norm, kstest, levy_stable
import optim.stable_estimators as se
import numpy as np
import pandas as pd
from scipy.stats import norm, levy_stable
import optim.stable_estimators as se
from tqdm import tqdm


# ÜBERARBEITEN MIT EIGENER FITTING METHODE 
def compare_stable_vs_gaussian(weights: np.ndarray):
    """
    1. fits a gaussian distribution to the input data
    2. fits a levy stable distribution to the input data

    perform a KS Test for both fits with the empirical data

    devide the p value of the stable fit by the p value of the gaussian fit: 
        r > 1 -> stable is a better fit
        r < 1 -> gaussian is a better fit

    Args:
        weights (np.ndarray): weight matrix of a layer
    """

    alpha, beta, sigma, mu = levy_stable.fit(weights.flatten())
    muN, sigN = norm.fit(weights.flatten())

    p_stable = kstest(weights.flatten(), 'levy_stable', args=(alpha, beta, sigma, mu))[1]
    p_gauss = kstest(weights.flatten(), 'norm', args=(muN, sigN))[1]

    r = p_stable / p_gauss

    return r


def eval_fit_methods(beta: float = 0, gamma: float = 1, delta: float = 0, n_samples: int = 10000, alpha_steps: int = 100, return_data: bool = False, verbose: bool = False):
    """
    Evaluates different alpha estimation methods on Levy stable distributions.
    
    Parameters
    ----------
    beta : float
        Skewness parameter.
    gamma : float
        Scale parameter.
    delta : float
        Location parameter.
    n_samples : int
        Number of samples to generate per alpha.
    alpha_steps : int
        Number of alpha values to test between 0.1 and 2.0.
    return_data : bool
        If True, return the last generated dataset as well.
    verbose: bool
        If True, shows progress while fitting 

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with true alpha and estimates from different methods.
    data : np.ndarray, optional
        The last generated dataset (if return_data=True)
    """
    alphas = np.linspace(0.1, 2.0, alpha_steps)
    results = []

    iterator = tqdm(alphas, desc="Estimating alpha") if verbose else alphas

    for alpha_val in iterator:
        # Generate data
        data = levy_stable.rvs(alpha_val, beta, loc=delta, scale=gamma, size=n_samples)

        # Fit using different methods, handle errors gracefully
        try:
            alpha_ml = se.maximum_likelihood_estimator(data)["alpha"]
        except Exception:
            alpha_ml = np.nan

        try:
            alpha_quantile = se.alpha_quantile_method(data)
        except Exception:
            alpha_quantile = np.nan

        try:
            alpha_logmom = se.alpha_log_moments(data)
        except Exception:
            alpha_logmom = np.nan

        try:
            alpha_tail = se.alpha_tail_regression(data)
        except Exception:
            alpha_tail = np.nan
        
        try: 
            alpha_robust = se.robust_alpha_estimator(data)
        except Exception:
            alpha_robust = np.nan

        results.append([alpha_val, alpha_ml, alpha_quantile, alpha_logmom, alpha_tail, alpha_robust])

    df_results = pd.DataFrame(results, columns=[
        "alpha_true", "alpha_ml", "alpha_quantile", "alpha_logmom", "alpha_tail", "alpha_robust"
    ])

    if return_data:
        return df_results, data
    return df_results
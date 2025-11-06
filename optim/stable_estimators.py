import numpy as np

def maximum_likelihood_estimator(data, verbose=0):
    """
    Estimate the parameters of a stable distribution using Maximum Likelihood Estimation (MLE). Due to the complexity of stable distributions, the function leverages R's 'fitstable' package via rpy2 for robust parameter estimation.
    
    Parameters:
        data (array-like): The input data to fit the stable distribution to.
        verbose (int): Verbosity level for R function output. Defaults to 0 (no output). Print "Parameters successfully estimated." if verbose = 1. Print estimated parameters if verbose = 2.
    
    Returns:
        dict: Estimated parameters (alpha, beta, gamma, delta).
    """

    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri, default_converter

    import rpy2.robjects as ro
    ro.r('.libPaths("/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library")')


    # activate conversion between R and pandas/numpy
    converter = default_converter + numpy2ri.converter + pandas2ri.converter
    robjects.conversion.set_conversion(converter)


    # load R script
    robjects.r['source'](r'/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/Bachelorarbeit/Code/optim/fit_stable.R')

    # call function 
    fit_stable_plot = robjects.globalenv['fit_stable_plot']

    result = fit_stable_plot(data, verbose)

    # convert the result to a python dictionary
    result_dict = {name: result[i][0] for i, name in enumerate(result.names())}
    result_dict = {str(k): float(v) for k, v in result_dict.items()}

    return result_dict

# ---------------------------
# The following methods only estimate alpha 
# ---------------------------

def alpha_quantile_method(data):
    """Estimate alpha using the quantile method.

    Parameters:
        data (array-like): The input data to fit the stable distribution to.
    
    Returns:
        floar: Alpha parameter estimate.
    """
    data0 = data - np.median(data)
    q5, q25, q50, q75, q95 = np.percentile(data0, [5, 25, 50, 75, 95])
    ratio = (q95 - q5) / (q75 - q25)

    if ratio < 4.0:
        alpha = 1.5
    elif ratio < 6.0:
        alpha = 1.8
    else:
        alpha = 2.0
    alpha = min(max(alpha, 0.1), 2.0)

    return alpha

def alpha_log_moments(data):
    """Estimate alpha using the log-moments method.

    Parameters:
        data (array-like): The input data to fit the stable distribution to.
    
    Returns:
        floar: Alpha parameter estimate.
    """
    data0 = np.abs(data - np.median(data))
    data0 = data0[data0 > 0]
    logx  = np.log(data0)
    m1 = np.mean(logx)
    m2 = np.mean(logx ** 2)

    alpha = np.pi / np.sqrt(6 * (m2 - m1**2))
    alpha = min(max(alpha, 0.1), 2.0)
    return alpha


def alpha_tail_regression(data, tail_fraction=0.1):
    """Estimate alpha using the tail-regression method.

    Parameters:
        data (array-like): The input data to fit the stable distribution to.
    
    Returns:
        floar: Alpha parameter estimate.
    """
    data0 = np.abs(data - np.median(data))
    data_sorted = np.sort(data0)
    n = len(data_sorted)
    tail_start = int((1.0 - tail_fraction) * n)
    tail_data  = data_sorted[tail_start:]

    tail_data = tail_data[tail_data > 0]
    log_vals = np.log(tail_data)
    ranks    = np.arange(1, len(tail_data) + 1)[::-1]
    log_ranks= np.log(ranks)
    A = np.vstack([log_vals, np.ones(len(log_vals))]).T
    slope, intercept = np.linalg.lstsq(A, log_ranks, rcond=None)[0]
    alpha = -slope
    alpha = min(max(alpha, 0.1), 2.0)
    return alpha

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri, default_converter
import rpy2.robjects as ro

def maximum_likelihood_estimator(data, verbose=0):
    """
    Estimates the parametres of a stable distribution using Maximum Likelihood Estimation. Due to the complexity of stable distributions, the function leverages R's 'fitstable' package via rpy2 for robust parameter estimation.
    
    Parameters:
        data (array-like): The input data to fit the stable distribution to.
        verbose (int): Verbosity level for R function output. Defaults to 0 (no output). Print "Parameters successfully estimated." if verbose = 1. Print estimated parameters if verbose = 2.
    
    Returns:
        dict: Estimated parameters (alpha, beta, gamma, delta).
    """

    # import numpy as np
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri, numpy2ri, default_converter

    # import rpy2.robjects as ro
    ro.r('.libPaths("/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library")')


    # activate conversion between R and pandas/numpy
    converter = default_converter + numpy2ri.converter + pandas2ri.converter
    robjects.conversion.set_conversion(converter)


    # load R script
    robjects.r['source'](r'/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/Bachelorarbeit/Code/estimator/fit_stable.R')

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

# def alpha_quantile_method(data): 
#     """Estimate alpha using the quantile method.

#     Parameters:
#         data (array-like): The input data to fit the stable distribution to.
    
#     Returns:
#         floar: Alpha parameter estimate.
#     """
#     data0 = data - np.median(data)
#     q5, q25, q50, q75, q95 = np.percentile(data0, [5, 25, 50, 75, 95])
#     ratio = (q95 - q5) / (q75 - q25)

#     if ratio < 4.0:
#         alpha = 1.5
#     elif ratio < 6.0:
#         alpha = 1.8
#     else:
#         alpha = 2.0
#     alpha = min(max(alpha, 0.1), 2.0)

#     return alpha

# import numpy as np


def alpha_quantile_method(data):
    """Estimate alpha using McCulloch's quantile method as in Nolan 2013.

    Parameters:
        data (array-like): Input data.
    
    Returns:
        float: Alpha estimate.
    """
    data0 = data - np.median(data)  
    q05, q25, q50, q75, q95 = np.percentile(data0, [5, 25, 50, 75, 95])
    

    ratio = (q95 - q05) / (q75 - q25)
    
    if ratio < 1.5:     
        alpha = 1.2
    elif ratio < 2.0:
        alpha = 1.5
    elif ratio < 2.5:
        alpha = 1.8
    else:
        alpha = 2.0
    return alpha

def alpha_hill_estimator(data, tail_fraction=0.1):
    """
    Estimate the tail index alpha using the classical Hill estimator.

    Parameters
    data : array-like
        Input data (assumed positive or heavy right tail).
    tail_fraction : float, optional
        Fraction of upper tail observations to use (default=0.1 → top 10%).

    Returns
    float
        Hill estimate of alpha (tail index).
    """
    data = np.asarray(data)
    data = np.abs(data - np.median(data))  
    data = data[data > 0]
    data_sorted = np.sort(data)[::-1]     
    n = len(data_sorted)
    k = max(1, int(tail_fraction * n))    

    topk = data_sorted[:k]
    xk1  = data_sorted[k] if k < n else data_sorted[-1]
    logs = np.log(topk) - np.log(xk1)
    hill = k / np.sum(logs)
    alpha = min(max(hill, 0.1), 2.0)
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


# def alpha_tail_regression(data, tail_fraction=0.1):
#     """Estimate alpha using the tail-regression method.

#     Parameters:
#         data (array-like): The input data to fit the stable distribution to.
    
#     Returns:
#         floar: Alpha parameter estimate.
#     """
#     data0 = np.abs(data - np.median(data))
#     data_sorted = np.sort(data0)
#     n = len(data_sorted)
#     tail_start = int((1.0 - tail_fraction) * n)
#     tail_data  = data_sorted[tail_start:]

#     tail_data = tail_data[tail_data > 0]
#     log_vals = np.log(tail_data)
#     ranks    = np.arange(1, len(tail_data) + 1)[::-1]
#     log_ranks= np.log(ranks)
#     A = np.vstack([log_vals, np.ones(len(log_vals))]).T
#     slope, intercept = np.linalg.lstsq(A, log_ranks, rcond=None)[0]
#     alpha = -slope
#     alpha = min(max(alpha, 0.1), 2.0)
#     return alpha

# def alpha_tail_regression(data, tail_fraction=0.2):
#     """Estimate alpha using tail regression (log-log Rank vs |X|); Nolan 

#     Parameters:
#         data (array-like): Input data.
#         tail_fraction (float): Fraction of upper tail to use.
    
#     Returns:
#         float: Alpha estimate.
#     """
#     data0 = np.asarray(data)
#     data0 = np.abs(data0 - np.median(data0)) 
#     data0 = data0[data0 > 0]
    
#     n = len(data0)
#     k = max(1, int(tail_fraction * n))
    
#     # us top k values
#     tail_data = np.sort(data0)[-k:]
    
#     # log-log Regression: log(Rank) ~ log(Tail Values)
#     ranks = np.arange(1, k + 1)[::-1]
#     log_ranks = np.log(ranks)
#     log_vals = np.log(tail_data)
    
#     A = np.vstack([log_vals, np.ones(k)]).T
#     slope, intercept = np.linalg.lstsq(A, log_ranks, rcond=None)[0]
#     alpha = -slope
#     alpha = min(max(alpha, 0.1), 2.0)
#     return alpha

def robust_alpha_estimator(data, tail_fraction=0.1):
    """
    Robust estimator combining tail-regression and MLE.
    
    - For alpha <= 0.5, use tail regression.
    - Otherwise, use MLE.
    - If MLE estimate > 2, set alpha = 2.
    """
    # Tail estimate
    alpha_tail = alpha_hill_estimator(data, tail_fraction=tail_fraction)
    
    if alpha_tail <= 0.5:
        return alpha_tail
    else:
        alpha_mle = maximum_likelihood_estimator(data)['alpha']
        alpha_mle = min(abs(alpha_mle), 2.0)
        return alpha_mle


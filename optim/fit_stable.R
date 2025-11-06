

options(repos = c(CRAN = "https://cloud.r-project.org"))


fit_stable_plot <- function(data, verbose=0) {

    suppressPackageStartupMessages({
    library(fBasics)
    library(stabledist)
    })

    # estimate parameters
    fit <- fBasics::stableFit(data, doplot = FALSE)

    # get parameters
    alpha_hat <- fit@fit$estimate["alpha"]
    beta_hat  <- fit@fit$estimate["beta"]
    gamma_hat <- fit@fit$estimate["gamma"]
    delta_hat <- fit@fit$estimate["delta"]

    if (verbose >= 1) {
            cat("Parameters successfully estimated\n")
        }
    if (verbose == 2) {
        cat(sprintf("alpha: %.4f\n", alpha_hat))
        cat(sprintf("beta : %.4f\n", beta_hat))
        cat(sprintf("gamma: %.4f\n", gamma_hat))
        cat(sprintf("delta: %.4f\n", delta_hat))
    }
    
    return(list(
    alpha = alpha_hat,
    beta  = beta_hat,
    gamma = gamma_hat,
    delta = delta_hat
    ))
}



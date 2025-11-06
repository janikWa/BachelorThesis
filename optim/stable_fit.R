# Installiere fBasics, falls nötig
if (!require(fBasics)) {
  install.packages("fBasics")
  library(fBasics)
}

# Beispielhafte stabile Verteilung simulieren
if (!require(stabledist)) {
  install.packages("stabledist")
  library(stabledist)
}

set.seed(123)
n <- 100000
alpha <- 1.9
beta <- 0.5
gamma <- 1
delta <- 0
data <- rstable(n, alpha=alpha, beta=beta, gamma=gamma, delta=delta, pm=0)

# Parameter fitten mit fBasics
fit <- fBasics::stableFit(data)

# Parameter aus dem S4-Objekt extrahieren
alpha_hat <- fit@fit$estimate["alpha"]
beta_hat  <- fit@fit$estimate["beta"]
gamma_hat <- fit@fit$estimate["gamma"]
delta_hat <- fit@fit$estimate["delta"]

# Histogramm vorbereiten
hist_data <- data.frame(x = data)

# Dichte der gefitteten Verteilung berechnen
x_vals <- seq(min(data), max(data), length.out = 1000)
y_vals <- dstable(x_vals, alpha=alpha_hat, beta=beta_hat, gamma=gamma_hat, delta=delta_hat, pm=0)
dens_data <- data.frame(x = x_vals, y = y_vals)

# Plot
library(ggplot2)
ggplot(hist_data, aes(x = x)) +
  geom_histogram(aes(y=..density..), bins = 100, fill = "lightblue", color = "black") +
  geom_line(data = dens_data, aes(x = x, y = y), color = "red", size = 1) +
  labs(title = "Histogramm der simulierten stabilen Verteilung mit gefitteter Dichte",
       x = "x", y = "Dichte") +
  theme_minimal()
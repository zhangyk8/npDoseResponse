# ``npDoseResponse``: An R Package for Nonparametric Inference on Dose-Response Curves Without the Positivity Condition

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/npDoseResponse)](https://CRAN.R-project.org/package=npDoseResponse)
<!-- badges: end -->

## Installation

The latest release of the R package can be installed through CRAN:

```R
install.packages("npDoseResponse")
```

The development version can be installed from github:

```R
devtools::install_github("zhangyk8/npDoseResponse", subdir = "R_Package")
```

## Toy Example

```R
require(npDoseResponse)
require(parallel)

set.seed(123)
n <- 300

S2 <- cbind(2 * runif(n) - 1, 2 * runif(n) - 1)
Z2 <- 4 * S2[, 1] + S2[, 2]
E2 <- 0.2 * runif(n) - 0.1
T2 <- cos(pi * Z2^3) + Z2 / 4 + E2
Y2 <- T2^2 + T2 + 10 * Z2 + rnorm(n, mean = 0, sd = 1)
X2 <- cbind(T2, S2)

t_qry2 = seq(min(T2) + 0.01, max(T2) - 0.01, length.out = 100)

theta_est2 = DerivEffect(Y2, X2, t_eval = t_qry2, h_bar = NULL, 
                          kernT_bar = "gaussian", h = NULL, b = NULL, 
                          C_h = 7, C_b = 3, print_bw = FALSE, 
                        degree = 2, deriv_ord = 1, kernT = "epanechnikov", 
                        kernS = "epanechnikov", parallel = TRUE, cores = 6)
                        
## This chunk of code could be time-consuming.               
# theta_boot2 = DerivEffectBoot(Y2, X2, t_eval = t_qry2, boot_num = 500, alpha=0.95,
#                        h_bar = NULL, kernT_bar = "gaussian", h = NULL, b = NULL,
#                        C_h = 7, C_b = 3, print_bw = FALSE, degree = 2,
#                        deriv_ord = 1, kernT = "epanechnikov", kernS = "epanechnikov",
#                        parallel = TRUE, cores = 6)

plot(t_qry2, theta_est2, type="l", col = "blue", xlab = "t", lwd=5, ylab="(Estimated) derivative effects")
lines(t_qry2, 2*t_qry2 + 1, col = "red", lwd=3)
legend(-2, 5, legend=c("Estimated derivative", "True derivative"),  fill = c("blue","red"))

m_est2 = IntegEst(Y2, X2, t_eval = t_qry2, h_bar = NULL, 
                  kernT_bar = "gaussian", h = NULL, b = NULL, 
                  C_h = 7, C_b = 3, print_bw = FALSE, 
                  degree = 2, deriv_ord = 1, kernT = "epanechnikov", 
                  kernS = "epanechnikov", parallel = TRUE, cores = 6)

## This chunk of code could be time-consuming.               
# m_boot2 = IntegEstBoot(Y2, X2, t_eval = t_qry2, boot_num = 500, alpha=0.95,
#                        h_bar = NULL, kernT_bar = "gaussian", h = NULL, b = NULL,
#                        C_h = 7, C_b = 3, print_bw = FALSE, degree = 2,
#                        deriv_ord = 1, kernT = "epanechnikov", kernS = "epanechnikov",
#                        parallel = TRUE, cores = 6)

plot(t_qry2, m_est2, type="l", col = "blue", xlab = "t", lwd=5, ylab="(Estimated) dose-response curves")
lines(t_qry2, t_qry2^2 + t_qry2, col = "red", lwd=3)
legend(-2, 6, legend=c("Estimated curve", "True curve"),  fill = c("blue","red"))
```


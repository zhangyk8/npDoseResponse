% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DerivEffect.R
\name{DerivEffect}
\alias{DerivEffect}
\title{The proposed localized derivative estimator.}
\usage{
DerivEffect(
  Y,
  X,
  t_eval = NULL,
  h_bar = NULL,
  kernT_bar = "gaussian",
  h = NULL,
  b = NULL,
  C_h = 7,
  C_b = 3,
  print_bw = TRUE,
  degree = 2,
  deriv_ord = 1,
  kernT = "epanechnikov",
  kernS = "epanechnikov",
  parallel = TRUE,
  cores = 6
)
}
\arguments{
\item{Y}{The input n-dimensional outcome variable vector.}

\item{X}{The input n*(d+1) matrix. The first column of X stores the
treatment/exposure variables, while the other d columns are confounding variables.}

\item{t_eval}{The m-dimensional vector for evaluating the derivative. (Default:
t_eval = NULL. Then, t_eval = \code{X[,1]}, which consists of the observed treatment variables.)}

\item{h_bar}{The bandwidth parameter for the Nadaraya-Watson conditional
CDF estimator. (Default: h_bar = NULL. Then, the Silverman's rule of thumb
is applied. See Chen et al. (2016) for details.)}

\item{kernT_bar}{The name of the kernel function for the Nadaraya-Watson conditional
CDF estimator. (Default: "gaussian".)}

\item{h, b}{The bandwidth parameters for the treatment/exposure variable
and confounding variables in the local polynomial regression.
(Default: h = NULL, b = NULL. Then, the rule-of-thumb bandwidth selector
in Eq. (A1) of Yang and Tschernig (1999) is used with additional scaling
factors C_h and C_b, respectively.)}

\item{C_h, C_b}{The scaling factors for the rule-of-thumb bandwidth parameters.}

\item{print_bw}{The indicator of whether the current bandwidth parameters
should be printed to the console. (Default: print_bw = TRUE.)}

\item{degree}{Degree of local polynomials. (Default: degree = 2.)}

\item{deriv_ord}{The order of the estimated derivative of the conditional mean
outcome function. (Default: deriv_ord = 1. It shouldn't be changed in most cases.)}

\item{kernT, kernS}{The names of kernel functions for the treatment/exposure
variable and confounding variables. (Default: kernT = "epanechnikov",
kernS = "epanechnikov".)}

\item{parallel}{The indicator of whether the function should be parallel
executed. (Default: parallel = TRUE.)}

\item{cores}{The number of cores for parallel execution. (Default: cores = 6.)}
}
\value{
The estimated derivative of the dose-response curve evaluated at
points \code{t_eval}.
}
\description{
This function implements our proposed estimator for estimating the derivative
of a dose-response curve via Nadaraya-Watson conditional CDF estimator.
}
\examples{
\donttest{
  library(parallel)
  set.seed(123)
  n <- 300

  S2 <- cbind(2 * runif(n) - 1, 2 * runif(n) - 1)
  Z2 <- 4 * S2[, 1] + S2[, 2]
  E2 <- 0.2 * runif(n) - 0.1
  T2 <- cos(pi * Z2^3) + Z2 / 4 + E2
  Y2 <- T2^2 + T2 + 10 * Z2 + rnorm(n, mean = 0, sd = 1)
  X2 <- cbind(T2, S2)

  t_qry2 = seq(min(T2) + 0.01, max(T2) - 0.01, length.out = 100)
  chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")
  if (nzchar(chk) && chk == "TRUE") {
    # use 2 cores in CRAN/Travis/AppVeyor
    num_workers <- 2L
  } else {
    # use all cores in devtools::test()
    num_workers <- parallel::detectCores()
  }
  theta_est2 = DerivEffect(Y2, X2, t_eval = t_qry2, h_bar = NULL,
                           kernT_bar = "gaussian", h = NULL, b = NULL,
                           C_h = 7, C_b = 3, print_bw = FALSE,
                           degree = 2, deriv_ord = 1, kernT = "epanechnikov",
                           kernS = "epanechnikov", parallel = TRUE, cores = num_workers)
  plot(t_qry2, theta_est2, type="l", col = "blue", xlab = "t", lwd=5,
       ylab="(Estimated) derivative effects")
  lines(t_qry2, 2*t_qry2 + 1, col = "red", lwd=3)
  legend(-2, 5, legend=c("Estimated derivative", "True derivative"),
         fill = c("blue","red"))
}

}
\references{
Zhang, Y., Chen, Y.-C., and Giessing, A. (2024)
\emph{Nonparametric Inference on Dose-Response Curves Without the Positivity Condition.}
\url{https://arxiv.org/abs/2405.09003}.

Hall, P., Wolff, R. C., and Yao, Q. (1999) \emph{Methods for Estimating A
Conditional Distribution Function. Journal of the American Statistical Association,
94 (445): 154-163}.
}
\author{
Yikun Zhang, \email{yikunzhang@foxmail.com}
}
\keyword{a}
\keyword{curve}
\keyword{derivative}
\keyword{dose-response}
\keyword{estimator}
\keyword{localized}
\keyword{of}
\keyword{the}

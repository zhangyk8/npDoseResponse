DerivEffect <- function(Y, X, t_eval = NULL, h_bar = NULL, kernT_bar = "gaussian", 
                        h = NULL, b = NULL, C_h = 7, C_b = 3, print_bw = TRUE, 
                        degree = 2, deriv_ord = 1, kernT = "epanechnikov", 
                        kernS = "epanechnikov", parallel = TRUE, cores = 6) {
  if (is.null(t_eval)) {
    t_eval <- X[,1]
  }
  
  n <- nrow(X)
  d <- 1
  if (is.null(h_bar)) {
    h_bar <- (4 / (d + 2))^(1 / (d + 4)) * (n^(-1 / (d + 4))) * sd(X[,1]) * sqrt((n-1)/n)
  }
  
  if (print_bw) {
    cat("The current bandwidth for the conditional CDF estimator is", h_bar, ".\n")
  }
  
  kernel_result <- KernelRetrieval(kernT_bar)
  kernT_bar <- kernel_result$KernFunc
  
  weight_mat <- kernT_bar((outer(t_eval, X[,1], "-")) / h_bar)
  weight_mat <- t(weight_mat / rowSums(weight_mat))
  weight_mat[is.nan(weight_mat)] <- 0
  
  if (parallel) {
    # Use parallel computing
    beta_mat <- mclapply(t_eval, function(t) {
      X_mat <- as.matrix(cbind(rep(t, n), X[, -1]))
      return(LocalPolyReg(Y, X, x_eval = X_mat, degree = degree, deriv_ord = deriv_ord, 
                          h = h, b = b, C_h = C_h, C_b = C_b, print_bw = print_bw, 
                          kernT = kernT, kernS = kernS))
    }, mc.cores = cores)
    beta_mat <- do.call(cbind, beta_mat)
  } else {
    beta_mat <- matrix(0, nrow = n, ncol = length(t_eval))
    for (i in 1:length(t_eval)) {
      X_mat <- as.matrix(cbind(rep(t_eval[i], n), X[, -1]))
      beta_mat[, i] <- LocalPolyReg(Y, X, x_eval = X_mat, degree = degree, 
                                    deriv_ord = deriv_ord, h = h, b = b, 
                                    C_h = C_h, C_b = C_b, print_bw = print_bw, 
                                    kernT = kernT, kernS = kernS)
    }
  }
  
  theta_C <- colSums(weight_mat * beta_mat)
  return(theta_C)
}

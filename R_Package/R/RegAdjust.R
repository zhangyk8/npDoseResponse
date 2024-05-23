RegAdjust <- function(Y, X, t_eval = NULL, h = NULL, b = NULL, C_h = 7, C_b = 3, 
                      print_bw = TRUE, degree = 2, deriv_ord = 0, 
                      kernT = "epanechnikov", kernS = "epanechnikov", 
                      parallel = FALSE, cores = 4) {
  
  if (is.null(t_eval)) {
    t_eval <- X[, 1]
  }
  
  n <- nrow(X)
  
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
  
  m_est <- colMeans(beta_mat)
  return(m_est)
}

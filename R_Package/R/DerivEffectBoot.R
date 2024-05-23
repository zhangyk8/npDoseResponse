DerivEffectBoot <- function(Y, X, t_eval = NULL, h_bar = NULL, kernT_bar = "gaussian", 
                            h = NULL, b = NULL, C_h = 7, C_b = 3, print_bw = TRUE, 
                            degree = 2, deriv_ord = 1, kernT = "epanechnikov", 
                            kernS = "epanechnikov", boot_num = 500, parallel = TRUE, 
                            cores = 6) {
  if (is.null(t_eval)) {
    t_eval <- X[, 1]
  }
  
  n <- nrow(X)
  
  theta_C_boot <- matrix(0, nrow = boot_num, ncol = length(t_eval))
  b <- 1
  while (b <= boot_num) {
    ind <- sample(n, n, replace = TRUE)
    X_boot <- X[ind,]
    Y_boot <- Y[ind]
    theta_C_boot[b, ] <- DerivEffect(Y_boot, X_boot, t_eval = t_eval, h_bar = h_bar, 
                                     kernT_bar = kernT_bar, h = h, b = b, C_h = C_h, 
                                     C_b = C_b, print_bw = print_bw, degree = degree, 
                                     deriv_ord = deriv_ord, kernT = kernT, kernS = kernS, 
                                     parallel = parallel, cores = cores)
    if (sum(is.nan(theta_C_boot[b, ])) == 0) {
      b <- b + 1
    }
  }
  
  return(theta_C_boot)
}

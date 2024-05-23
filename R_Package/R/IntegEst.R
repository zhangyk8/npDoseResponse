IntegEst <- function(Y, X, t_eval = NULL, h_bar = NULL, kernT_bar = "gaussian", 
                     h = NULL, b = NULL, C_h = 7, C_b = 3, print_bw = TRUE, 
                     degree = 2, deriv_ord = 1, kernT = "epanechnikov", 
                     kernS = "epanechnikov", parallel = TRUE, cores = 6) {
  if (is.null(t_eval)) {
    t_eval <- X[, 1]
  }
  
  T_sort <- sort(X[, 1])
  n <- nrow(X)
  
  theta_est <- DerivEffect(Y, X, t_eval = T_sort, h_bar = h_bar, kernT_bar = kernT_bar, 
                           h = h, b = b, C_h = C_h, C_b = C_b, print_bw = print_bw, 
                           degree = degree, deriv_ord = deriv_ord, kernT = kernT, 
                           kernS = kernS, parallel = parallel, cores = cores)
  
  T_delta <- T_sort[2:n] - T_sort[1:(n-1)]
  
  int_mat_up <- matrix(T_delta * (1:(n-1)) * theta_est[1:(n-1)], nrow = n-1, ncol = n)
  int_mat_up <- int_mat_up * outer(1:(n-1), 1:n, "<")
  
  int_mat_down <- matrix(T_delta * (n - 1:(n-1)) * theta_est[2:n], nrow = n-1, ncol = n)
  int_mat_down <- int_mat_down * outer(1:(n-1), 1:n, ">=")
  
  m_samp <- mean(Y) + colSums(int_mat_up - int_mat_down) / n
  
  m_est <- approx(T_sort, m_samp, xout = t_eval, method = "linear")$y
  
  return(m_est)
}

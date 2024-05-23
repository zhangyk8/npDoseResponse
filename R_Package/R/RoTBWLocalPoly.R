RoTBWLocalPoly <- function(Y, X, kernT = "epanechnikov", kernS = "epanechnikov", 
                           C_h = 7, C_b = 3) {
  n <- nrow(X)
  d <- ncol(X) - 1
  
  kernel_result <- KernelRetrieval(kernT)
  kernT <- kernel_result$KernFunc
  sigmaK_sq <- kernel_result$sigmaK_sq
  K_sq <- kernel_result$K_sq
  
  p_coeff <- unname(lm(Y ~ poly(X[,1], 4, raw = TRUE))$coefficients)
  sec_deriv <- 12 * p_coeff[5] * X[,1] + 6 * p_coeff[4] * X[,1] + 2 * X[,1]
  C_fun <- mean(sec_deriv^2)
  
  T1 <- matrix(X[,1], ncol = 1)
  lhs <- as.matrix(cbind(1, X, T1^2, T1^3, T1^4))
  rcond <- .Machine$double.eps * max(dim(lhs))
  beta <- solve(t(lhs) %*% lhs, t(lhs) %*% Y, tol = rcond)
  
  resid <- sum((Y - lhs %*% beta)^2) * (max(X[,1]) - min(X[,1])) / (n - 5)
  sigmaK_sq <- sigmaK_sq^2
  
  h <- ((K_sq * resid) / (4 * n * sigmaK_sq * C_fun))^(1/5) * (n^(d / (5 * (d + 5)))) * C_h
  
  kernel_result <- KernelRetrieval(kernS)
  kernS <- kernel_result$KernFunc
  sigmaK_sq <- kernel_result$sigmaK_sq
  K_sq <- kernel_result$K_sq
  
  sec_deriv <- matrix(0, nrow = n, ncol = d)
  for (i in 2:(d + 1)) {
    p_coeff <- unname(lm(Y ~ poly(X[,i], 4, raw = TRUE))$coefficients)
    sec_deriv[,i-1] <- 12 * p_coeff[5] * X[,i] + 6 * p_coeff[4] * X[,i] + 2 * X[,i]
  }
  C_fun <- sum(diag(t(sec_deriv) %*% sec_deriv)) / n
  
  lhs <- as.matrix(cbind(1, X, T1^2, T1^3, T1^4))
  rcond <- .Machine$double.eps
  beta <- solve(t(lhs) %*% lhs, t(lhs) %*% Y, tol = rcond)
  S_mat = as.matrix(X[,2:(d+1)])
  resid <- sum((Y - lhs %*% beta)^2) * (apply(S_mat, 2, max) - apply(S_mat, 2, min)) / (n-5)
  
  sigmaK_sq <- sigmaK_sq^2
  K_sq <- K_sq^d
  b <- ((K_sq * d * resid) / (4 * n * sigmaK_sq * C_fun))^(1 / (d + 5)) * C_b
  
  return(list(h = h, b = b))
}

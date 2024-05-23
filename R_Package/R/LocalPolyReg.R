LocalPolyReg <- function(Y, X, x_eval = NULL, degree = 2, deriv_ord = 1, h = NULL, 
                         b = NULL, C_h = 7, C_b = 3, print_bw = TRUE, 
                         kernT = "epanechnikov", kernS = "epanechnikov") {
  if (is.null(x_eval)) {
    x_eval <- X
  }
  
  if (is.null(h) && is.null(b)) {
    bw_params <- RoTBWLocalPoly(Y, X, kernT = kernT, kernS = kernS, C_h = C_h, C_b = C_b)
    h <- bw_params$h
    b <- bw_params$b
  } else if (is.null(h)) {
    bw_params <- RoTBWLocalPoly(Y, X, kernT = kernT, kernS = kernS, C_h = C_h, C_b = C_b)
    h <- bw_params$h
  } else if (is.null(b)) {
    bw_params <- RoTBWLocalPoly(Y, X, kernT = kernT, kernS = kernS, C_h = C_h, C_b = C_b)
    b <- bw_params$b
  }
  
  if (print_bw) {
    cat("The current bandwidth for treatment variable in the local polynomial regression is", h, ".\n")
    cat("The current bandwidth for confounding variables in the local polynomial regression is", b, ".\n")
  }
  
  Y_est <- LocalPolyRegMain(Y, X, x_eval = x_eval, degree = degree, deriv_ord = deriv_ord, 
                            h = h, b = b, kernT = kernT, kernS = kernS)
  return(Y_est)
}

KernelRetrieval <- function(name) {
  if (name == "rectangular") {
    return(list(KernFunc = rectangular, sigmaK_sq = 1/3, K_sq = 1/2))
  }
  
  if (name == "triangular") {
    return(list(KernFunc = triangular, sigmaK_sq = 1/6, K_sq = 2/3))
  }
  
  if (name == "epanechnikov") {
    return(list(KernFunc = epanechnikov, sigmaK_sq = 1/5, K_sq = 3/5))
  }
  
  if (name == "biweight") {
    return(list(KernFunc = biweight, sigmaK_sq = 1/7, K_sq = 5/7))
  }
  
  if (name == "triweight") {
    return(list(KernFunc = triweight, sigmaK_sq = 1/9, K_sq = 350/429))
  }
  
  if (name == "tricube") {
    return(list(KernFunc = tricube, sigmaK_sq = 35/243, K_sq = 175/247))
  }
  
  if (name == "gaussian") {
    return(list(KernFunc = gaussian, sigmaK_sq = 1, K_sq = 1/(2*sqrt(pi))))
  }
  
  if (name == "bigaussian") {
    return(list(KernFunc = bigaussian, sigmaK_sq = 3/2, K_sq = 3*sqrt(2/pi)/8))
  }
  
  if (name == "cosine") {
    return(list(KernFunc = cosine, sigmaK_sq = 1 - 8/(pi^2), K_sq = pi^2/16))
  }
  
  if (name == "logistic") {
    return(list(KernFunc = logistic, sigmaK_sq = pi^2/3, K_sq = 1/6))
  }
  
  if (name == "sigmoid") {
    return(list(KernFunc = sigmoid, sigmaK_sq = pi^2/4, K_sq = 2/(pi^2)))
  }
  
  if (name == "silverman") {
    return(list(KernFunc = silverman, sigmaK_sq = 0, K_sq = 3*sqrt(2)/16))
  }
  
  return(NULL)
}

rectangular <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs(0.5 * ind)
  return(res)
}

triangular <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs((1 - abs(t)) * ind)
  return(res)
}

epanechnikov <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs(0.75 * (1 - t^2) * ind)
  return(res)
}

biweight <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs(((15/16) * (1 - t^2)^2) * ind)
  return(res)
}

triweight <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs((35/32) * (1 - t^2)^3 * ind)
  return(res)
}

tricube <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs((70/81) * (1 - abs(t)^3)^3 * ind)
  return(res)
}

gaussian <- function(t) {
  res <- (1 / sqrt(2 * pi)) * exp(-0.5 * t^2)
  return(res)
}

bigaussian <- function(t) {
  res <- (2 / sqrt(pi)) * (t^2) * exp(-t^2)
  return(res)
}

cosine <- function(t) {
  ind <- (abs(t) <= 1)
  res <- abs((pi/4) * cos(pi * t / 2) * ind)
  return(res)
}

logistic <- function(t) {
  res <- 1 / (exp(t) + 2 + exp(-t))
  return(res)
}

sigmoid <- function(t) {
  res <- (2 / pi) / (exp(t) + exp(-t))
  return(res)
}

silverman <- function(t) {
  res <- 0.5 * exp(-abs(t) / sqrt(2)) * sin(abs(t) / sqrt(2) + pi / 4)
  return(res)
}
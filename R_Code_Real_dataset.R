\begin{verbatim}
## Load required packages
library(np)          # nonparametric regression
library(dplyr)
library(MASS)

## Read data directly from UCI
abalone <- read.table(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/
  abalone/abalone.data",
  sep = ","
)

colnames(abalone) <- c(
  "Sex", "Length", "Diameter", "Height",
  "WholeWeight", "ShuckedWeight", "VisceraWeight",
  "ShellWeight", "Rings"
)

## Remove categorical variable
abalone <- abalone %>% select(-Sex)

## Standardize continuous covariates
abalone_scaled <- as.data.frame(scale(abalone))
## Response
Y <- abalone_scaled$Rings

## Parametric covariates (including intercept)
X <- cbind(
  1,
  abalone_scaled$Length,
  abalone_scaled$Diameter
)

colnames(X) <- c("Intercept", "Length", "Diameter")

## Nonparametric covariates
V <- abalone_scaled %>%
  select(WholeWeight, ShuckedWeight, ShellWeight)
## Step 1: Nonparametric regression of Y on V
bw_y <- npregbw(xdat = V, ydat = Y)
mY_hat <- fitted(npreg(bw_y))

## Step 2: Nonparametric regression of X on V (column-wise)
mX_hat <- apply(X[, -1], 2, function(xj) {
  bw_x <- npregbw(xdat = V, ydat = xj)
  fitted(npreg(bw_x))
})

## Step 3: Partial residuals
Y_tilde <- Y - mY_hat
X_tilde <- X[, -1] - mX_hat

## Step 4: Estimate beta via least squares
beta_hat 
<- solve(t(X_tilde) %*% X_tilde) %*% t(X_tilde) %*% Y_tilde
beta_hat
## Parametric component
A <- X[, -1] %*% beta_hat

## Nonparametric component
B <- mY_hat
## Empirical marginal CDFs
F_A <- ecdf(A)
F_B <- ecdf(B)

## Empirical joint CDF
F_AB <- function(a, b) {
  mean(A <= a & B <= b)
}
gridA <- quantile(A, probs = seq(0.1, 0.9, by = 0.2))
gridB <- quantile(B, probs = seq(0.1, 0.9, by = 0.2))

joint_vals 
<- outer(gridA, gridB, Vectorize(F_AB))
prod_vals 
<- outer(gridA, gridB, function(a, b) F_A(a) * F_B(b))

max(abs(joint_vals - prod_vals))
\end{verbatim}
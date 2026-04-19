```r
# ==========================================================
# REAL DATA APPLICATION:
# GPLR MODEL WITH JOINT DISTRIBUTION OF
# PARAMETRIC AND NONPARAMETRIC COMPONENTS
# ==========================================================

# Required Packages
library(readr)
library(dplyr)
library(ggplot2)
library(np)
library(KernSmooth)
library(MASS)
library(mgcv)
library(plot3D)
library(gridExtra)

# ==========================================================
# STEP 1: LOAD THE DATASET
# ==========================================================

# Replace with your actual file path
heart_data <- read_csv("heart_attack_prediction_dataset.csv")

# Display structure
str(heart_data)
summary(heart_data)

# ==========================================================
# STEP 2: DATA PREPROCESSING
# ==========================================================

# Convert categorical variables to numeric indicators
heart_data <- heart_data %>%
  mutate(
    Smoking = ifelse(Smoking == "Yes", 1, 0),
    Diabetes = ifelse(Diabetes == "Yes", 1, 0),
    FamilyHistory = ifelse(FamilyHistory == "Yes", 1, 0),
    PreviousHeartProblem = ifelse(PreviousHeartProblems == "Yes", 1, 0),
    Sex = ifelse(Sex == "Male", 1, 0)
  )

# Construct a continuous cardiovascular burden score
heart_data <- heart_data %>%
  mutate(
    RiskScore = 0.03*Age +
      0.60*Smoking +
      0.55*Diabetes +
      0.45*FamilyHistory +
      0.70*PreviousHeartProblem +
      0.01*Cholesterol +
      0.02*BMI +
      0.03*StressLevel
  )

# Response variable
Y <- heart_data$RiskScore

# ==========================================================
# STEP 3: DEFINE PARAMETRIC AND NONPARAMETRIC COVARIATES
# ==========================================================

# Parametric component covariates
X <- as.matrix(heart_data[, c(
  "Age",
  "Smoking",
  "Diabetes",
  "FamilyHistory",
  "PreviousHeartProblem"
)])

# Nonparametric component covariates
V <- as.matrix(heart_data[, c(
  "Cholesterol",
  "BMI",
  "StressLevel",
  "ExerciseHoursPerWeek",
  "SedentaryHoursPerDay"
)])

# ==========================================================
# STEP 4: ESTIMATE CONDITIONAL EXPECTATIONS
# Robinson-type partial residual estimation
# ==========================================================

n <- nrow(X)
p <- ncol(X)

# Estimate E(Y|V)
EY_V <- npreg(
  txdat = data.frame(V),
  tydat = Y
)$mean

# Estimate E(X_j|V) for each parametric covariate
EX_V <- matrix(0, n, p)

for(j in 1:p){
  EX_V[,j] <- npreg(
    txdat = data.frame(V),
    tydat = X[,j]
  )$mean
}

# Centered quantities
Y_tilde <- Y - EY_V
X_tilde <- X - EX_V

# ==========================================================
# STEP 5: ESTIMATE BETA HAT
# ==========================================================

beta_hat <- solve(t(X_tilde) %*% X_tilde) %*% t(X_tilde) %*% Y_tilde
beta_hat

# Estimated parametric component
A_hat <- as.vector(X %*% beta_hat)

# ==========================================================
# STEP 6: ESTIMATE NONPARAMETRIC COMPONENT
# ==========================================================

# Residuals after removing parametric effect
residuals_np <- Y - A_hat

# Fit nonparametric model using generalized additive model
np_model <- gam(
  residuals_np ~
    s(Cholesterol) +
    s(BMI) +
    s(StressLevel) +
    s(ExerciseHoursPerWeek) +
    s(SedentaryHoursPerDay),
  data = heart_data
)

summary(np_model)

# Estimated nonparametric component
B_hat <- predict(np_model)

# Fitted values
Y_hat <- A_hat + B_hat

# ==========================================================
# STEP 7: JOINT DISTRIBUTION ESTIMATION
# ==========================================================

# Empirical marginal distribution functions
F_A <- ecdf(A_hat)
F_B <- ecdf(B_hat)

# Grid values
s1_grid <- seq(min(A_hat), max(A_hat), length.out = 50)
s2_grid <- seq(min(B_hat), max(B_hat), length.out = 50)

# Joint empirical distribution matrix
joint_cdf <- matrix(0, nrow = length(s1_grid), ncol = length(s2_grid))
product_marginals <- matrix(0, nrow = length(s1_grid), ncol = length(s2_grid))

for(i in 1:length(s1_grid)){
  for(j in 1:length(s2_grid)){
    joint_cdf[i,j] <- mean(A_hat <= s1_grid[i] & B_hat <= s2_grid[j])
    product_marginals[i,j] <- F_A(s1_grid[i]) * F_B(s2_grid[j])
  }
}

# Difference matrix
joint_difference <- abs(joint_cdf - product_marginals)

# ==========================================================
# STEP 8: TABLE OF JOINT VS PRODUCT OF MARGINALS
# ==========================================================

comparison_table <- data.frame(
  s1 = c(quantile(A_hat, 0.25), quantile(A_hat, 0.50), quantile(A_hat, 0.75), quantile(A_hat, 0.90)),
  s2 = c(quantile(B_hat, 0.25), quantile(B_hat, 0.50), quantile(B_hat, 0.75), quantile(B_hat, 0.90))
)

comparison_table$JointCDF <- mapply(function(a,b){
  mean(A_hat <= a & B_hat <= b)
}, comparison_table$s1, comparison_table$s2)

comparison_table$ProductMarginals <- mapply(function(a,b){
  F_A(a) * F_B(b)
}, comparison_table$s1, comparison_table$s2)

comparison_table$AbsoluteDifference <- abs(
  comparison_table$JointCDF - comparison_table$ProductMarginals
)

print(comparison_table)

# ==========================================================
# STEP 9: RISK CLASSIFICATION
# ==========================================================

A_cut <- median(A_hat)
B_cut <- median(B_hat)

risk_group <- ifelse(A_hat <= A_cut & B_hat <= B_cut, "Low Risk",
                     ifelse(A_hat > A_cut & B_hat <= B_cut, "Moderate Risk: Linear",
                            ifelse(A_hat <= A_cut & B_hat > B_cut, "Moderate Risk: Nonlinear",
                                   "Very High Risk")))

heart_data$RiskGroup <- risk_group

table(heart_data$RiskGroup)

# ==========================================================
# STEP 10: PLOT OF PARAMETRIC VS NONPARAMETRIC COMPONENTS
# ==========================================================

plot1 <- ggplot(data.frame(A_hat, B_hat, RiskGroup = risk_group),
                aes(x = A_hat, y = B_hat, color = RiskGroup)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "Joint Distribution of Estimated Components",
    x = expression(A[i] == X[i]^T * hat(beta)),
    y = expression(B[i] == hat(m)(V[i]))
  ) +
  theme_minimal()

print(plot1)

# ==========================================================
# STEP 11: NONLINEAR EFFECT OF CHOLESTEROL
# ==========================================================

chol_grid <- seq(min(heart_data$Cholesterol),
                 max(heart_data$Cholesterol),
                 length.out = 100)

newdata_chol <- data.frame(
  Cholesterol = chol_grid,
  BMI = mean(heart_data$BMI, na.rm = TRUE),
  StressLevel = mean(heart_data$StressLevel, na.rm = TRUE),
  ExerciseHoursPerWeek = mean(heart_data$ExerciseHoursPerWeek, na.rm = TRUE),
  SedentaryHoursPerDay = mean(heart_data$SedentaryHoursPerDay, na.rm = TRUE)
)

chol_effect <- predict(np_model, newdata = newdata_chol)

plot2 <- ggplot(data.frame(Cholesterol = chol_grid,
                           Effect = chol_effect),
                aes(x = Cholesterol, y = Effect)) +
  geom_line(size = 1.2, color = "blue") +
  labs(
    title = "Estimated Nonlinear Effect of Cholesterol",
    x = "Cholesterol",
    y = "Estimated Nonparametric Effect"
  ) +
  theme_minimal()

print(plot2)

# ==========================================================
# STEP 12: HEATMAP OF JOINT DISTRIBUTION DIFFERENCE
# ==========================================================

heatmap_data <- expand.grid(
  s1 = s1_grid,
  s2 = s2_grid
)

heatmap_data$Difference <- as.vector(joint_difference)

plot3 <- ggplot(heatmap_data, aes(x = s1, y = s2, fill = Difference)) +
  geom_tile() +
  labs(
    title = "Absolute Difference Between Joint CDF and Product of Marginals",
    x = expression(s[1]),
    y = expression(s[2])
  ) +
  theme_minimal()

print(plot3)

# ==========================================================
# STEP 13: SAVE OUTPUTS
# ==========================================================

write.csv(comparison_table,
          "joint_distribution_comparison_table.csv",
          row.names = FALSE)

write.csv(data.frame(beta_hat),
          "estimated_beta_coefficients.csv",
          row.names = TRUE)

write.csv(data.frame(
  A_hat = A_hat,
  B_hat = B_hat,
  RiskGroup = risk_group
),
"estimated_components.csv",
row.names = FALSE)

# Save plots
ggsave("joint_distribution_scatter.png", plot1, width = 8, height = 6)
ggsave("cholesterol_effect.png", plot2, width = 8, height = 6)
ggsave("joint_difference_heatmap.png", plot3, width = 8, height = 6)

# ==========================================================
# END OF SCRIPT
# ==========================================================
```

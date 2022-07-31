Data Modeling
================

-   <a href="#1-introduction" id="toc-1-introduction">1. Introduction</a>
    -   <a href="#11-loading-packages" id="toc-11-loading-packages">1.1 Loading
        Packages</a>
    -   <a href="#12-importing-data" id="toc-12-importing-data">1.2 Importing
        Data</a>
-   <a href="#2-data-preparation" id="toc-2-data-preparation">2. Data
    Preparation</a>
    -   <a href="#21-data-cleaning" id="toc-21-data-cleaning">2.1 Data
        Cleaning</a>
    -   <a href="#22-feature-engineering" id="toc-22-feature-engineering">2.2
        Feature Engineering</a>
    -   <a href="#23-validation-set" id="toc-23-validation-set">2.3 Validation
        Set</a>
-   <a href="#3-logistic-regression-model"
    id="toc-3-logistic-regression-model">3. Logistic Regression Model</a>
    -   <a href="#31-building-the-model" id="toc-31-building-the-model">3.1
        Building the Model</a>
    -   <a href="#32-calculating-model-performance"
        id="toc-32-calculating-model-performance">3.2 Calculating Model
        Performance</a>
-   <a href="#4-improving-the-model" id="toc-4-improving-the-model">4.
    Improving the Model</a>
    -   <a href="#41-determining-the-optimal-cutoff-point"
        id="toc-41-determining-the-optimal-cutoff-point">4.1 Determining the
        Optimal Cutoff Point</a>
    -   <a href="#42-bootstrapping" id="toc-42-bootstrapping">4.2
        Bootstrapping</a>
    -   <a href="#43-plotting-model-performance"
        id="toc-43-plotting-model-performance">4.3 Plotting Model
        Performance</a>
-   <a href="#5-discussion" id="toc-5-discussion">5. Discussion</a>
    -   <a href="#51-feature-importance" id="toc-51-feature-importance">5.1
        Feature Importance</a>
    -   <a href="#52-model-comparison" id="toc-52-model-comparison">5.2 Model
        Comparison</a>
    -   <a href="#53-other-ways-to-improve-performance"
        id="toc-53-other-ways-to-improve-performance">5.3 Other Ways to Improve
        Performance</a>
-   <a href="#6-conclusion" id="toc-6-conclusion">6. Conclusion</a>

## 1. Introduction

This section will focus on building a logistic regression model to
predict customer churning based on my findings from the EDA section and
measure its performance using the testing set. I will also attempt to
improve the model’s performance using methods such as bootstrapping.

**Key Findings:**

-   The initial logistic regression model had an accuracy of 79%, but a
    sensitivity of 46.9%. This means that over half of churning
    customers were incorrectly classified.
-   By changing the probability threshold from 0.5 to 0.219, I was able
    to increase the sensitivity to 80.7% at the cost of a 13.1% decrease
    in accuracy. This translates to a 72% increase in the number of
    churning customers who are correctly classified.
-   By using bootstrapping, I increased sensitivity to 72% at the cost
    of a 9.4% decrease in accuracy.
-   Depending on the cost of customer retention and customer loss, each
    model has its own advantages and disadvantages.
-   The type of contract a customer has is the most important variable
    in predicting churning. Other important variables include their
    monthly charge, payment method, and whether or not they have certain
    services provided by the company.

### 1.1 Loading Packages

``` r
# Load required libraries
library(dplyr)
library(plyr)
library(reshape2)
library(caret)
library(cutpointr)

# Set seed for reproducible results
set.seed(101)
```

### 1.2 Importing Data

``` r
# Import data
load("data/train.Rdata")
load("data/test.Rdata")
```

## 2. Data Preparation

Before building the model, I need to prepare the dataset based on my
findings from the EDA section. This includes handling missing values and
feature engineering.

### 2.1 Data Cleaning

Since I imported the raw data again, I need to perform the same data
cleaning I did in the EDA section to both the training and testing sets.

``` r
# Drop the first column, which contains customer ID
train <- subset(train, select = -c(customerID))
test <- subset(test, select = -c(customerID))

# Drop the missing values
train <- train[complete.cases(train), ]
test <- test[complete.cases(test), ]

# Change senior citizen to a factor variable
train$SeniorCitizen <- as.factor(mapvalues(train$SeniorCitizen, from=c("0","1"), to=c("No", "Yes")))
test$SeniorCitizen <- as.factor(mapvalues(test$SeniorCitizen, from=c("0","1"), to=c("No", "Yes")))
```

### 2.2 Feature Engineering

Next, I created 3 new features using the variables in the dataset:

1.  **MonthlyContract** - whether or not a customer has a
    *month-to-month* contract
2.  **HasService** - whether or not a customer has at least one of
    variables: online security, online backup, device protection, tech
    support
3.  **ECheck** - whether or not a customer’s payment method is
    *electronic check*

I verified the results by comparing their counts to the original
variables or viewing sample rows of the data.

``` r
# Recode contract, payment method, and the offered services to create the new features
train$MonthlyContract <- as.factor(ifelse(train$Contract == "Month-to-month", "Yes", "No"))

train$HasService <- as.factor(ifelse(train$OnlineSecurity == "Yes", "Yes",
                              ifelse(train$OnlineBackup == "Yes", "Yes",
                              ifelse(train$DeviceProtection == "Yes", "Yes",
                              ifelse(train$TechSupport == "Yes", "Yes", "No")))))

train$ECheck <- as.factor(ifelse(train$PaymentMethod == "Electronic check", "Yes", "No"))

# Verify the new features by comapring counts
summary(subset(train, select = c("MonthlyContract", "Contract")))
```

    ##  MonthlyContract           Contract   
    ##  No : 897        Month-to-month:1098  
    ##  Yes:1098        One year      : 433  
    ##                  Two year      : 464

``` r
summary(subset(train, select = c("ECheck", "PaymentMethod")))
```

    ##  ECheck                       PaymentMethod
    ##  No :1312   Bank transfer (automatic):449  
    ##  Yes: 683   Credit card (automatic)  :415  
    ##             Electronic check         :683  
    ##             Mailed check             :448

``` r
# Verify the new feature by viewing sample rows of the dataset
sample_n(subset(train, select = c("HasService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport")), 5)
```

    ##   HasService      OnlineSecurity        OnlineBackup    DeviceProtection
    ## 1        Yes                  No                 Yes                 Yes
    ## 2         No No internet service No internet service No internet service
    ## 3         No No internet service No internet service No internet service
    ## 4        Yes                  No                  No                 Yes
    ## 5        Yes                  No                  No                 Yes
    ##           TechSupport
    ## 1                 Yes
    ## 2 No internet service
    ## 3 No internet service
    ## 4                  No
    ## 5                  No

Once I verified the changes, I created the features in the testing set.

``` r
# Create features in the testing set
test$MonthlyContract <- as.factor(ifelse(test$Contract == "Month-to-month", "Yes", "No"))

test$HasService <- as.factor(ifelse(test$OnlineSecurity == "Yes", "Yes",
                              ifelse(test$OnlineBackup == "Yes", "Yes",
                              ifelse(test$DeviceProtection == "Yes", "Yes",
                              ifelse(test$TechSupport == "Yes", "Yes", "No")))))

test$ECheck <- as.factor(ifelse(test$PaymentMethod == "Electronic check", "Yes", "No"))
```

### 2.3 Validation Set

Finally, I split half the testing set to create a validation set. The
validation set will be reserved to measure the model performance of the
final models.

``` r
# Split the test set in half to create a test set and a validation set
sample <- sample(nrow(test), nrow(test) * 0.5, replace = FALSE)
val.set <- test[sample, ]
test <- test[-sample, ]
```

## 3. Logistic Regression Model

The logistic regression model involves modeling the log-odds ratio of a
customer churning. For a more detailed description of the method and why
I used it, see my blog post.

### 3.1 Building the Model

I fit a logistic regression model on **churning** and the 5 variables I
determined to be significant from the EDA section.

The predictors are:

1.  Monthly Contract
2.  Monthly Charges
3.  Dependents
4.  Electronic Check
5.  Has Service

``` r
# Fit logistic regression model
log.fit <- glm(Churn ~ MonthlyContract + MonthlyCharges + Dependents + ECheck + HasService, data = train, family = binomial)

# View summary of the logistic regression model
summary(log.fit)
```

    ## 
    ## Call:
    ## glm(formula = Churn ~ MonthlyContract + MonthlyCharges + Dependents + 
    ##     ECheck + HasService, family = binomial, data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6866  -0.7060  -0.3432   0.8706   2.6498  
    ## 
    ## Coefficients:
    ##                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)        -3.736816   0.220898 -16.917  < 2e-16 ***
    ## MonthlyContractYes  2.001968   0.152383  13.138  < 2e-16 ***
    ## MonthlyCharges      0.023288   0.002608   8.931  < 2e-16 ***
    ## DependentsYes      -0.212779   0.144954  -1.468    0.142    
    ## ECheckYes           0.654802   0.121681   5.381 7.40e-08 ***
    ## HasServiceYes      -0.719699   0.142819  -5.039 4.67e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2337.8  on 1994  degrees of freedom
    ## Residual deviance: 1810.0  on 1989  degrees of freedom
    ## AIC: 1822
    ## 
    ## Number of Fisher Scoring iterations: 5

**Summary of Findings:**

-   The estimated coefficient for **monthly contract** is 2.002, which
    means the log-odds ratio of churning is 2.002 greater if a customer
    has a month-to-month contract.
-   The estimated coefficient for **monthly charges** is 0.023, which
    means for every $1 increase in the monthly subscription cost, the
    log-odds ratio of churning increases by 0.023.
-   The estimated coefficient for **dependents** is -0.213, which means
    the log-odds ratio of churning is 0.213 less if a customer has
    dependents.
-   The estimated coefficient for **electronic check** is 0.655, which
    means the log-odds ratio of churning is 0.655 greater is a customer
    uses electronic checks as their payment method
-   The estimated coefficient for **has service** is -0.720, which means
    the log-odds ratio of churning is 0.720 less if a customer has at
    least one of the following: online security, online backup, device
    protection, or tech support.
-   All estimated coefficients are significant except for
    **dependents**.

### 3.2 Calculating Model Performance

To calculate model performance, I defined a function that will fit the
logistic regression model on the testing and obtain the confusion
matrix. This confusion matrix will be used to calculate the performance
metrics such as accuracy, sensitivity, and specificity.

``` r
confusion.matrix <- function(model, k, data) {
  
  # Obtain fitted probabilities from the test data
  glm.prob <- predict(model, type = "response", newdata = data)
  
  # Obtain vector of predictions using a k-value threshold
  glm.pred <- rep("No", nrow(data))
  glm.pred[glm.prob >= k] = "Yes"
  
  # Obtain confusion matrix and metrics
  cmat <- table(glm.pred, data$Churn)
  metrics <- confusionMatrix(cmat, positive = "Yes")
  
  # Display class counts
  print(summary(data$Churn))
  
  # Display confusion matrix
  print(metrics$table)
  
  # Print out accuracy, sensitivity, and specificity measures
  cat("\nAccuracy: ", round(metrics$overall["Accuracy"], 3))
  cat("\nSensitivity: ", round(metrics$byClass["Sensitivity"], 3))
  cat("\nSpecificity: ", round(metrics$byClass["Specificity"], 3))
  
  return(metrics)
}
```

``` r
# Obtain the confusion matrix of the model with a 0.5 probability threshold
log.matrix <- confusion.matrix(log.fit, 0.5, test)
```

    ##  No Yes 
    ## 772 226 
    ##         
    ## glm.pred  No Yes
    ##      No  682 120
    ##      Yes  90 106
    ## 
    ## Accuracy:  0.79
    ## Sensitivity:  0.469
    ## Specificity:  0.883

Although the model has an accuracy of 79%, the sensitivity is only
46.9%. This means that out of the 226 churning customers in the testing
set, more than half, (120 customers), have been incorrectly classified
as not churning. This can be costly to the company if they fail to
target customers who are likely to leave. Thus, adjustments to the model
must be made.

## 4. Improving the Model

### 4.1 Determining the Optimal Cutoff Point

One way to improve the model is to change the probability threshold, or
cutoff point, when classifying customers as churning or not. To identify
the optimal cutoff point, I plotted a ROC curve of the model using the
predicted probabilities, which graphs the true positive rate
(Sensitivity) against the false positive rate (1 - Specificity).

``` r
# Store predicted probabilities in the dataframe
glm.prob <- predict(log.fit, type = "response", newdata = test)
test$prob <- glm.prob

# Using the predicted probabilities, construct a ROC Curve
cp <- cutpointr(test, prob, Churn, pos_class = "Yes", neg_class = "No", direction = ">=",
                method = maximize_metric, metric = sum_sens_spec)

# Summarize the results of the ROC curve
summary(cp)
```

    ## Method: maximize_metric 
    ## Predictor: prob 
    ## Outcome: Churn 
    ## Direction: >= 
    ## 
    ##     AUC   n n_pos n_neg
    ##  0.8156 998   226   772
    ## 
    ##  optimal_cutpoint sum_sens_spec    acc sensitivity specificity  tp fn  fp  tn
    ##            0.2186        1.4907 0.6884      0.8496      0.6412 192 34 277 495
    ## 
    ## Predictor summary: 
    ##     Data       Min.         5%    1st Qu.    Median      Mean   3rd Qu.
    ##  Overall 0.01989301 0.02991060 0.06022692 0.2147470 0.2663675 0.4483929
    ##       No 0.01989301 0.02982794 0.04440649 0.1316928 0.2108171 0.3343328
    ##      Yes 0.05610163 0.12096634 0.29417438 0.4898122 0.4561236 0.6278435
    ##        95%      Max.        SD NAs
    ##  0.6639751 0.7674626 0.2198025   0
    ##  0.6336011 0.7613819 0.1946797   0
    ##  0.7173003 0.7674626 0.1935309   0

``` r
# Visualize the ROC curve
plot(cp)
```

![](telco_model_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

The optimal cutoff value was determined to be 0.283. This means that if
the model’s predicted probability of churning is 0.283 or more, the
customer will be classified as churning.

The cutoff point can be located in the upper left hand corner of the ROC
curve, where it attempts to maximize sensitivity and specificity. The
plots to the show the distribution of predicted probabilities under each
class. As you can see, the majority of churning customers are correctly
classified.

By changing the optimal cutoff value, the sensitivity increased to 0.850
at the cost of a lower specificity at 0.641. Depending on the costs of
customer retention strategies and the cost of losing a customer, the
trade off between sensitivity and specificity could be worth it.

``` r
# Obtain the optimal cutoff value from the ROC curve
k <- cp$optimal_cutpoint

# Construct a new confusion matrix with the optimal cutoff
log.matrix2 <- confusion.matrix(log.fit, k, test)
```

    ##  No Yes 
    ## 772 226 
    ##         
    ## glm.pred  No Yes
    ##      No  495  34
    ##      Yes 277 192
    ## 
    ## Accuracy:  0.688
    ## Sensitivity:  0.85
    ## Specificity:  0.641

192 out of 226 churning customers have been correctly classified as
churning, which is an increase of 72 customers when using a 0.5 cutoff.
However, 277 out of the 772 non-churning customers have been incorrectly
classified as churning, which is an increase of 187 customers.

### 4.2 Bootstrapping

Another way of improving model performance is bootstrapping.
Bootstrapping involves sampling from the available data with replacement
(i.e., oversampling). I will sample enough observations from each class
of customers until I have an equal number of customers who churned and
who did not churn.

``` r
# Split the training data by churning
train.churn <-  subset(train, Churn == "Yes")
train.not.churn <-  subset(train, Churn == "No")

# Sample from each set by the number of customers in the opposite set
churn.sample <- sample(nrow(train.churn), nrow(train.not.churn), replace = TRUE)
not.churn.sample <- sample(nrow(train.not.churn), nrow(train.churn), replace = TRUE)

# Add the bootstrapped samples to the training set
train.boot <- rbind(train, train.churn[churn.sample,], train.not.churn[not.churn.sample,])

# Verify that the prevalence of churning is 50%
summary(train.boot$Churn)
```

    ##   No  Yes 
    ## 1995 1995

Once I have my bootstrapped samples, I will fit a logistic regression
model on the data.

``` r
# Fit logistic regression model on the bootstrapped dataset
log.fit.boot <- glm(Churn ~ MonthlyContract + MonthlyCharges + Dependents + ECheck + HasService, data = train.boot, family = binomial)

# View summary of the resulting logistic regression model
summary(log.fit.boot)
```

    ## 
    ## Call:
    ## glm(formula = Churn ~ MonthlyContract + MonthlyCharges + Dependents + 
    ##     ECheck + HasService, family = binomial, data = train.boot)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -2.11364  -0.82682   0.08564   0.79888   2.33467  
    ## 
    ## Coefficients:
    ##                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)        -2.819621   0.135476 -20.813  < 2e-16 ***
    ## MonthlyContractYes  1.977105   0.090787  21.777  < 2e-16 ***
    ## MonthlyCharges      0.024893   0.001679  14.829  < 2e-16 ***
    ## DependentsYes      -0.339537   0.093007  -3.651 0.000262 ***
    ## ECheckYes           0.583148   0.081205   7.181 6.91e-13 ***
    ## HasServiceYes      -0.681595   0.095447  -7.141 9.26e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5531.3  on 3989  degrees of freedom
    ## Residual deviance: 4172.3  on 3984  degrees of freedom
    ## AIC: 4184.3
    ## 
    ## Number of Fisher Scoring iterations: 4

The resulting coefficients are interpreted the same way as the initial
model. In fact, many of the coefficients are similar to the initial
model. In this model, however, the **dependents** is significant.

``` r
# Calculate the confusion matrix
boot.matrix <- confusion.matrix(log.fit.boot, 0.5, test)
```

    ##  No Yes 
    ## 772 226 
    ##         
    ## glm.pred  No Yes
    ##      No  552  54
    ##      Yes 220 172
    ## 
    ## Accuracy:  0.725
    ## Sensitivity:  0.761
    ## Specificity:  0.715

The bootstrapped model has an accuracy of 72.5%, a sensitivity of 76.1%,
and a specificity of 71.5%. Compared to the first model, the sensitivity
is higher, but not as high as the sensitivity from changing the cutoff
value.

### 4.3 Plotting Model Performance

To measure the final model performance of the models, I will now use the
validation set, which acts as completely unseen data for the models.
This will be a good representation on how the models will perform on new
data.

``` r
log.matrix.val <- confusion.matrix(log.fit, k, val.set)
```

    ##  No Yes 
    ## 713 285 
    ##         
    ## glm.pred  No Yes
    ##      No  428  55
    ##      Yes 285 230
    ## 
    ## Accuracy:  0.659
    ## Sensitivity:  0.807
    ## Specificity:  0.6

``` r
boot.matrix.val <- confusion.matrix(log.fit.boot, 0.5, val.set)
```

    ##  No Yes 
    ## 713 285 
    ##         
    ## glm.pred  No Yes
    ##      No  489  79
    ##      Yes 224 206
    ## 
    ## Accuracy:  0.696
    ## Sensitivity:  0.723
    ## Specificity:  0.686

To easily compare the performance of all the models, I created a
function that will store the metrics into a dataframe.

``` r
# Define function for extracting desired performance metric measures into a dataframe
add.performance.metrics <- function(metrics, model) {
  accuracy <- round(metrics$overall["Accuracy"], 3)
  sensitivity <- round(metrics$byClass["Sensitivity"], 3)
  specificity <- round(metrics$byClass["Specificity"], 3)
  balanced.accuracy <- round(metrics$byClass["Balanced Accuracy"], 3)
  
  return(data.frame(model, accuracy, sensitivity, specificity, balanced.accuracy, row.names = NULL))
}

# Create empty dataframe of performance metrics
metrics.df <- data.frame(matrix(ncol = 5, nrow = 0), row.names = NULL)

# Rename columns
col <- c("model", "accuracy", "sensitivity", "specificity", "balanced_accuracy")
colnames(metrics.df) <- col

# Create a list of the calculated confusion matrices
matrices <- list(log.matrix, log.matrix2, log.matrix.val, boot.matrix, boot.matrix.val)

# Create a vector of model names
names <- c("1. Base Model", "2. Optimal Cutoff", "3. Optimal Cutoff (Val)", "4. Bootstrapped Model","5. Bootstrapped Model (Val)")

# Iterate through each confusion matrix and use function to obtain performance metrics
for (i in 1:length(matrices)) {
  df <- add.performance.metrics(matrices[[i]], names[i])
  metrics.df <- rbind(metrics.df, df)
}

# View performance metrics dataframe
metrics.df
```

    ##                         model accuracy sensitivity specificity
    ## 1               1. Base Model    0.790       0.469       0.883
    ## 2           2. Optimal Cutoff    0.688       0.850       0.641
    ## 3     3. Optimal Cutoff (Val)    0.659       0.807       0.600
    ## 4       4. Bootstrapped Model    0.725       0.761       0.715
    ## 5 5. Bootstrapped Model (Val)    0.696       0.723       0.686
    ##   balanced.accuracy
    ## 1             0.676
    ## 2             0.745
    ## 3             0.704
    ## 4             0.738
    ## 5             0.704

Then, I plotted the performance metrics of the models below.

``` r
# Expand dataframe to a long format
metrics.long <- melt(metrics.df, id.vars = "model", variable.name = "metrics")

# Plot metrics
ggplot(metrics.long, aes(model, value, group = metrics)) + 
  geom_point(aes(color = metrics), size = 2) + 
  geom_line(aes(color = metrics), size = 1) + 
  xlab("Model") + 
  ylab("Value") + 
  ggtitle("Model Performance") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 15)) 
```

![](telco_model_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

For each model, the results from the validation set is slightly lower
than the results from the testing set. The **balanced accuracy**, which
is the average of sensitivity and specificity, is equal for both models
at 70.4%.

Since the balanced accuracy of both models is equal, I decided to select
the bootstrapped as the final model due to its higher accuracy (69.6%
vs. 65.9%).

## 5. Discussion

### 5.1 Feature Importance

Feature importance measures how influential a predictor is in predicting
churning. One way of measuring influence is by looking at the
coeficients of the model.

``` r
# Obtain odds ratio for the coefficients
exp(log.fit.boot$coefficients)
```

    ##        (Intercept) MonthlyContractYes     MonthlyCharges      DependentsYes 
    ##         0.05962856         7.22180820         1.02520558         0.71209978 
    ##          ECheckYes      HasServiceYes 
    ##         1.79167057         0.50580960

Assuming all other remaining variables are constant:

-   Customers with a month-to-month contract have a **622% increase** in
    the odds of churning.
-   Customers with a dependent have a **28.8% decrease** in the odds of
    churning.
-   Customers who use electronic checks have a **79.2% increase** in the
    odds.
-   Customers that have one of those services have a **49.4% decrease**
    in the odds of churning.
-   For every $1 increase in monthly charge, the odds of churning
    **increases by 2.5%**.

All predictors appear to have a significant influence in churning. Based
on these numbers, **monthly contract** has the most influence in the
odds of churning while **dependents** has the least influence. Thus,
Telco Company can prioritize customers who have a month-to-month
contract.

### 5.2 Model Comparison

As previously stated, both models have an equal balanced accuracy of
70.4%. I decided to select the bootstrapped model as the final model due
to its higher accuracy (69.6% vs. 65.9%). However, the regular logistic
regression model (with the optimal cutoff point) could be the better
model due to its higher sensitivity (80.7% s 72.3%).

For example, if the loss of a customer is more costly than the cost of
any implemented customer retention strategies, then this model will be a
better option since it ensures that customers who are more likely to
churn are identified.

On the other hand, if the cost is about the same, then the bootstrapped
model will be a better option since it ensures that as many customers
are correctly classified regardless of whether or not they are more
likely to churn or not.

### 5.3 Other Ways to Improve Performance

Changing the probability threshold and bootstrapping both led to
improved models. However, there are other options I could implement to
improve the model.

First, I only selected 5 variables from the 19 potential predictors in
the dataset, so it might be worth revisiting the other variables. Also,
I there were some predictors chosen that were highly correlated to
another. For example, **internet services** is highly correlated with
**monthly charges**, so I could include that in the model instead of
monthly charges. The same is also true for **tenure** and **monthly
contract**.

Finally, there are other classification methods I can use such as random
forest to predict churning that might result in a better performance. I
could determine which methods will be appropriate to use and perform any
necessary hyperparameter tuning.

## 6. Conclusion

In this project, I managed to develop a logistic regression model that
predicts customers churning. I was able to increase the performance of
the model by changing the probability threshold of classification and by
using bootstrapping. Depending on the cost of implementing customer
retention strategies and the cost of losing a customer, each model has
its own advantages and disadvantages. Finally, by including only five
predictors, the model is easily interpretable with actionable insights
that can be derived from it, such as the type of customers to target and
prioritize.

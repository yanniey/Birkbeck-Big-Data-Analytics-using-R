# Big Data Analytics using R
## Master of Data Science at Birkbeck, University of London, 10/2018

### Course Format:
* On campus: 1.5 hours of lecture and 1.5 hours of lab per week
* Online: ~ 5 hours of coding coursework & assignments
* 2 courseworks of 10% each
* Exam: 3 hour exams (80% of score)


### Syllabus 
1. Introduction to R
2. Basic Statistics
3. Linear Regression
4. Logistic Regression
5. Cross Validation
6. Decision Trees
7. Ensemble Methods
8. SVM
9. Clustering
10. Dimension Reduction
11. Applications

## Week 1: Introduction to R
**rm(list=ls())** removes everything in the environment

## Week 2: Descriptive statistics
Descriptive Analysis

#### 1. Univariate vs. Bivariate

  * Single value vs. relationship between paris of variables 

#### 2. Variance vs. Standard Deviation
  1. Variance (s2(sample); σ2(population))
      * The average of the squared differences from the mean

  2. Standard Deviation (s(sample); σ(population)) 
      * Equals to square root of variance

#### 3. Biased vs. unbiased sample variance
* Biased sample variance is divided by **n**;
* **Unbiased** sample variance is divided by **n-1**;
* The biased sample variance (↓) usually underestimates the population variance (↑)
  + The observations of a sample fall, on average, closer to the sample mean than to the population mean
  + Using n-1 instead of n as the divisor corrects that by making the result a little bit bigger

#### 4. Covariance vs. correlation

  1. Covariance is a dimensional quantity
      * The value depends on the units of the data
      * Difficult to compare covariances among data sets that have different scales.
      * Covariance = Σ (Xi- Xavg)(Yi-Yavg) / N
      * use **N for population, n-1 for samples**
      * Covariance measures how much the movement in one variable predicts the movement in a corresponding variable
         * **Positive** covariance indicates that **greater** values of one variable tend to be paired with **greater** values of the other variable.
         * **Negative** covariance indicates that **greater** values of one variable tend to be paired with **lesser** values of the other variable.
  2. Correlation is a dimensionless quantity
      * Always between -1 and 1
      * Facilitates the comparison of different data sets
      *  correlation of X and Y =
covariance of X and Y /
(standard deviation of X * standard deviation of Y)




#### 5. In R
1. **var()** and **sd()** in R uses *n-1* as denominator. 
2. Interquartile Range is Q3 - Q1
3. **set.seed(m)** reproduces the exact same set of random numbers as long as the arbitrary integer argument m stays the same.
4. **rnorm(n,mean=0,sd=1)** generates a vector of random normal variables
  * n: sample size
  * default mean=0 and sd=1
  * each time different




## Week 3: Simple Linear Regression
Three big questions for this week:
1. How to estimate?
2. How to assess accuracy of the coefficient estimates (Hypothetis Testing)
3. How to assess accuracy of the model


#### Step 1: How to estimate:
1. Residual sum of squares
2. Least square line:  
    1. Sum of the residuals equals 0
    2. Residual sum of squares is minimised
    least square estimates 

3. How close is a single sample mean `Ƹ𝜇` to the population mean `μ`?  
    * Answer: Use standard error (SE): the average amount that this estimate `ො𝜇` differs from `μ`
    * The more observations we have, the smaller the Standard Error is     

#### Step 2: How to assess accuracy of the efficient estimates (Comparing coefficients only)

Three lines:  
1. True relationship line: `y = f(x)+ ε`
2. Population regression line: `y = b0 + b1x + ε`
   *Popluation regress line is the best **linear** approximation to the true relationship between X and Y
3. Least square line: `y(with hat) = 𝛽0(with hat) + 𝛽1(with hat)x`

#### Standard error (SE)
Standard error is the average amount that this estimate 𝜇-hat differs from μ.  

```SE= variance / n```

The more observations we have, the smaller the SE is. 

#### Hypothesis Tests
Is `B1 = 0` or not? If we can't be sure that B1 `!= then` there is no point using X as the predicto. 

1. **Null hypothesis** - There is no relationship between X and Y

2. **Alternative hypothesis** - There is some relationship between X and Y

3. The higher **t-value** is (equivalently p-value is small), the more possible X and Y are related, then we can be sure that B1(hat) is not 0.
  * We can reject the Null Hypothesis.
  * We declare a relationship to exist between X and Y.
  
4. **P-value**: P values evaluate how well the sample data support that the null hypothesis is true. It measures how compatible your data are with the null hypothesis
  * A smallp-value (typically ≤ 0.05) indicates yoursample provides strong evidence against the null hypothesis, so you reject the null hypothesis.
  * A largep-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
  * Typical p-value cutoffs for rejecting the null hypothesis are 5 or 1%  

5. Summary of t-value and p-value  
  * The t-test produces a single value,t, which grows larger as the difference between the means of two samples grows larger;  
  * t does not cover a fixed range such as 0 to 1 like probabilities do;
  * You can convert a t-value into a probability, called a p-value;  
  * The p-value is always between 0 and 1 and it tells you the probability of the difference in your data being due to sampling error;  
  * The p-value should be lower than a chosen significance level (0.05 for example) before you can reject your null hypothesis.


#### Step 3: How to assess accuracy of the model
1. Residual Standard Error (RSE)
  *  It measures the degress to which the model fits the data (e.g. should this be linear vs. non-linear regression?)
  *  RSE is the estimate of the standard deviation of `ε`
  *  RSE has a unit, and it has the same unit as `Y`
  *  Quantifies average amount that the response will deviate from the population regression

2. Measuring the extent to which the model fits the data  
  R<sup>2</sup> statistic
  * Some of the variation in Y can be explained by variation in the X’s and some cannot.
  * R<sup>2</sup> tells you the **proportion of variance** of Y that can be explained by X.
  * **The higher the R<sup>2</sup>, the better fit the model is.**
  * In simple linear regression, R<sup>2</sup> == Cor(X,Y)<sup>2</sup>

3. `Cor(X,Y) = 0` means there is no **linear** relationship between X and Y, but there could be other relationship.

4. Multiple R-squared and adjusted R-squared:
  * Model with multiple variables: use adjusted R-squared 
  * Model with single variable: use R squared and adjusted R squared interchangably 
  * Generally **use adjusted R-squared** as preference

5. Predicting mean is a lot more accurate than predicting an individual

6. **Prediction interval (PI)** is an estimate of an interval in which future observations (particular individuals) will fall, with a certain probability, given what has already been observed.

7. **Confidence intervals** predict **confidence intervals** for the **mean**, **prediction intervals** for the **individuals**.

Confidence interval is between (the lines are lower than) prediction interval.

#### Conclusion
1. Practical Interpretation of `y-intercept`:
* predictedyvalue when x= 0–no practical interpretation if x= 0 is either nonsensical or outside range of sample data


#### Assignment in Week 3's presentation:
```{r}
fert = c(4,6,10,12)
yield=c(3.0,5.5,6.5,9.0)
FertYield=data.frame(fert,yield)
fert.fit<-lm(yield~fert,data=FertYield)
predict(fert.fit,data.frame(a=(c(2.5,5.5,8.5))),interval="confidence")
confint(fert.fit)
summary(fert.fit)
```


## Week 4 Logistic Regression
Regression vs. Classification

#### Step 1. Come up with `b0 hat` and `b1 hat` to estimate `b0` and `b1` (i.e. How to estimate coefficients)

* Use **maximum likelihood**

We try to find `b0 hat`  and `b1 hat` such that plugging these estimates into
yields
  + a number close to `1` for all individuals who chose Innocent, and
  + a number close to `0` for all individuals who did not choose Innocent.
  + use likelihood function to maximise likelihood 

* Interpreting `b1`

  + If `b1` = 0, this means that there is no relationship between Y and X.
  + If `b1` > 0, this means that when X gets larger so does the probability that `Y = 1`, that is, X and Pr(Y =1) are <u>positively correlated</u>.


#### Step 2. Are the coefficients significant? (Hypothesis Test)
* Is `b1` equal to 0? - null hypothesis
  + In linear regression, we use `t-test`: the higher `t-value` is, the better it is.
  + In logistic regression, we use `z-test`: the higher `z-value` is, the better. 
  + `p-value` is interpreted the same way:
    + If `p-value` is tiny, reject `H0` → there is relationship between `X` and `Pr(Y=1)`
    + Otherwise, accept `H0` → there is no relationship between `X` and `Pr(Y=1)`

+ How sure are we about our guesses for `b0` and `b1`?

#### Step 3. How to make predictions?

#### Case Study: Credit Card Default Data

Qualitative Predictors in Logistic Regression

  * We can predict if an individual default by checking if s/he is a student or not. Thus we can use a qualitative variable “student” coded as (Student = 1, Non-student =0). How?
  መ
  * `b` is positive: This indicates students tend to have higher default probabilities than non-students








## Week 5: Assessing Model Accuracy & Cross Validation
1. How to assess model accuracy
  * For a **regression** problem, we used the **MSE** to assess the accuracy of the statistical learning method
  * For a **classification** problem we can use the **error rate**(e.g. ConfusionMatrix).

2. **Residual sum of squares(RSS)**: The sum of the squares of residuals. It is a measure of the discrepancy between the data and an estimation model. A small RSS indicates a tight fit of the model to the data.

3. A common measure of accuracy is the **mean squared error (MSE)**, which is equal to `1/n * RSS`

4. Our method has generally been designed to make MSE small on the training data we are looking at:
  * e.g. with linear regression we choose the line such that MSE (RSS) is minimisedàleast squares line.

5. Training vs Test MSE

  * In general, the **more flexible** a method is, the lower its **training MSE** will be.
  * However, the test MSE may in fact be higher for a flexible method than a simple approach (e.g. linear regression)

6. Bias/Variance tradeoff
  * **Bias** refers to the error that is introduced by modeling a real life problem (that is usually extremely complicated) by a much simpler problem.
    + The more flexible/complex a method is the less bias it will generally have.
  * **Variance** refers to how much your estimate for f would change by if you had a different training data set (from the same population).
    + The more flexible a method is the more variance it has.

7. Expected test MSE
  * `ExpectedTestMSE = Bias^2 +Var+ σ^2`. `σ^2` is irreducible error. 
  * We want to minimise `ExpectedTestMSE`.
  * It is a challenge to find a method for which both the variance and the squared bias are low.

8. How to calculate MSE in R?
  Consider the linear regression models, this calculates the training MSE:
  ```r
  lm.fit <- lm(y~x,data=DS) 
  mean((y-predict(lm.fit,DS))^2)
  ```

9. For classification models, we'll use Confusion Matrix/Error Rate

  `Accuracy = # correct predictions / total # of predictions`
  `Error rate = # wrong predictions / total # of predictions`

10. How to Calculate Error Rate in R?  
  In logistic regression, calculate the training error rate
  * Building the `glm.fit`
  * Using `glm.fit` to make probability predictions
  * Set a threshold (could be 0.5, or other number) to make qualitative
  predictions based on the probability predictions
  * Using `table()` function to build a confusion matrix
  * Using `mean()` function to calculate the error rate


  ```r
  glm.fit <- glm(default~balance,data = Default, family=binomial) dim(Default)
  #[1] 10000 4
  glm.probs <- predict(glm.fit, Default, type="response") 
  glm.pred <- rep("Yes",10000)
  glm.pred[glm.probs<.5] <- "No" table(glm.pred,default)
  default
  glm.pred 

      No Yes
  No 9625 233 
  Yes 42 100


  # Accuracy
  mean(glm.pred==default)
  [1] 0.9725
  # Error rate
  mean(glm.pred!=default)
  [1] 0.0275
  ```

11. Cross Validation
  * Estimate the test error rate by
    + holding out a subset of the training observations from the fitting process, and then
    + applying the statistical learning method to those held out observations.

  * Cross Validation on Regression Problems
    + The Validation Set Approach
    + Leave-One-Out Cross Validation
    + K-fold Cross Validation
      + Bias-Variance Trade-off for k-fold Cross Validation

 * Cross Validation on Classification Problems

12. **Validation Set Approach**: randomly spliting the data into training part and validation(testing, or hold-out) part

  How to do it?
  * Randomly split Auto data set (392 obs.) into training (196 obs.) and validation data (196 obs.)
    ```r
    set.seed(1)
    train <- sample(392,196)
    ```

  * Fit the model using the training data set

  ```r
  lm.fit.train <- lm(mpg~horsepower,data=Auto,subset=train)
  ```

  * Then, evaluate the model using the validation data set

  ```r
    mean((Auto$mpg-predict(lm.fit.train,Auto))[-train]^2) 
    [1] 26.14142
  ```

  Plot the observations and linear relationship between mpg and horsepower:

  ```r
    plot(Auto$horsepower, Auto$mpg, xlab="horsepower",
    ylab="mpg",
    col="blue") abline(lm.fit.train,col="red")
  ```

  Improving the model using **quadratic** model. Evaluate the model using validation data set([-train] means that we will only use the test dataset): 
  ```r
    set.seed(1)
    train <- sample(392,196)
    lm.fit2.train <- lm(mpg~poly(horsepower,2),data=Auto, subset=train)
    mean((Auto$mpg-predict(lm.fit2.train,Auto))[-train]^2) 
    [1] 19.82259 
    #linear model's MSE: 26.14142
  ```

  The quadratic model has a smaller test error, thus it is better than the simple linear regression model. 

  If test MSEs are very different from each other, it means that this model has high variance. 

  * The Validation Set Approach
  
    + Advantages:
      + Simple
      + Easy to implement

    + Disadvantages:
      + The validation MSE can be highly variable
      + Only a subset of observations are used to fit the model (training data). Statistical methods tend to perform worse when trained on fewer observations.

13. **Leave-One-Out Cross Validation (LOOCV)**

  * This method is similar to the Validation Set Approach, but it tries to address the latter's disadvantages. 
  * For each suggested model, do:
    + Split the data set of size `n` into:
      + training data set size: `n-1`
      + testing data set size: `1`
    + Fit the model using training data set 
    + Validate model using the testing data, and compute the corresponding MSE
    + Reeat this process `n` times
    + Calculate the average of the MSEs = `1/n * MSE`

  LOOCV vs. Validation Set Approach
  * Advantage: LOOCV has less bias
    + We repeatedly fit the statistical learning method using training data that contains `n - 1` obs., i.e. almost all the data set is used
  * Advantage LOOCV produces a less variable MSE 
    + The validation set approach produces different MSE when applied repeatedly due to randomness in the splitting process
    + Performing LOOCV multiple times will always yield the same results, because we split based on 1 obs. each time
  * Disadvantage: LOOCV is computationally intensive 
    + We fit a model n times!

   How to do LOOCV in R:
   Using the Auto data set again, building a linear model 
   
   ```r
    glm.fit <- glm(mpg~horsepower,data=Auto)
    # This is the same as lm(mpg~horsepower,data=Auto)
    library(boot) 
    #cv.glm() is in the boot library
    cv.err <- cv.glm(Auto,glm.fit)
    # cv.glm() does the LOOCV
    cv.err$delta
    [1] 24.23151 24.23114 #(raw CV est., adjusted CV est.) 
    #The MSE is 24.23114 (use the adjusted CV value).
   ```

14. **K-fold Cross Validation**

  * LOOCV is computationally intensive, so we can run k-fold Cross Validation instead
  * With k-fold CV, we divide the data set into k different parts (e.g. k = 5, or k = 10, etc.)
  * We then remove the first part, fit the model on the remaining k-1 parts, and see how good the predictions are on the left out part (i.e. compute the MSE on the first part)
  * We then repeat this k different times taking out a different part each time
  * By averaging the k different MSE’s, we get an estimated validation (test) error rate for new observations

  How to do K-fold validation in R:
  ```r
    glm.fit <- glm(mpg~horsepower,data=Auto)
    # This is the same as in LOOCV
    library(boot)  # This is the same as in LOOCV
    cv.err <- cv.glm(Auto,glm.fit, K=10)
    #K means K-fold, can be 5, 10 or other numbers
    cv.err$delta
    [1] 24.3120 24.2926
  ```  
  The MSE is `24.3120`.

15. Comparing LOOCV and K-fold Cross validation
  * LOOCV is **less bias** than k-fold CV (when k < n)
    + LOOCV: uses n-1 observations
    + K-fold CV: uses (k-1)n/k observations
  * But, LOOCV has **higher variance** than k-fold CV (when k < n)
    + The mean of many highly correlated quantities has higher variance
  * Thus, there is a trade-off between what to use
  * Conclusion: we usually use K-fold (k=5 or k=10)

16. Cross validation for classification problems
  We can use cross validation in a classification situation in a similar manner
  * Divide data into k parts
  * Hold out one part, fit using the remaining data and compute the **error rate** on the held out data
  * Repeat k times
  * CV error rate is the average over the k errors we have computed

## Week 5: Decision Trees
Regression vs. Classification tree

#### Regression tree: Base ball player salary set  

1. Why did we transform the salary from numbers to `log numbers`?
  * There are significant difference between salaries -> from tens to thousands.
  * We log transformed the salary because the difference between min and max values have big gaps. We use log transform to make the numbers to normal distribution (because there are way more tools to deal with normal distribution than non-normal distribution)

  * When there's a large gap between min and max value, then often the statistical tools won't perform well when doing analysis.

  * Log changes multiplication into addition (for which there are more tools)

2. Where to split?
  * Cutting point = the average between points
  * Stopping criteria: stop when there are too few observations in each area (e.g. < 5 observations>)

3. Improving tree accuracy
  * A large tree (i.e. one with many terminal nodes) may tend to over fit the training data.
    + Large tree: lower bias, higher variance, worse interpretation 
    + Small tree: higher bias, lower variance, better interpretation
  * Generally, we can improve accuracy by “pruning” the tree i.e. cutting off some of the terminal nodes.
  * How do we know how far back to prune the tree?
    + We use cross validation to see which tree has the **lowest error rate**.

#### Classification tree:
* For each region (or node) we predict the most common category among the training data within that region.
* There are several possible different criteria to use such as the “gini index” and “cross-entropy” but the easiest one to think about is to minimise the **error rate**.
* `Error rate = # wrong predictions / total # of predictions`
* `tree.carseats <- tree(High~.-Sales,Carseats)`
  `.-Sales` means x is everything but `Sales`. removing it because `High` is correlated with `Sales`

How to prune for different trees:
  * `prune.misclass` for classification tree
  * `prune.tree` for regression tree

## Week 6. Ensemble Methods (Decision Trees, Random Forest/Bagging/Boosting)
Decision trees have high variance (a disadvantage)

1. Pros of Decision Trees:
  + Trees are very easy to explain to people (probably even easier than linear regression)
  + Treescanbeplottedgraphically
  + They work fine on both classification and regression problems

2. Cons of Decision Trees:
  + Trees don’t have the same prediction accuracy as some of the more complicated approaches that we examine in this course
  + **High variance**

3. Bootstraping 
  * Resampling of the observed dataset (and of equal size to the observed dataset), each of which is obtained by **random sampling with replacement** from the original dataset.
  * Distinct test sets are usually there to obtain a measure of **variability** – how the test MSE/error rate varies

4. Bagging = Bootstrapping + averaging
  * Bootstraping = plenty of training datasets
  * Averaging= reduces variances
  * How does Bagging work?
    + How does bagging work?
    + Generate B different bootstrapped training datasets
    + Train the statistical learning method (e.g. a decision tree) on each of the B training datasets, and obtain the prediction
  * For prediction:
    + **Regression**: average all predictions from all B trees
    + **Classification**: majority vote among all B trees
5. Pruned trees = lower variance, 
   Unpruned tress=  high variance  but low bias
   Averaging trees reduce variance and bias, but loses interpretability. 

6. Bagging for regression vs. classification trees
   For prediction for classification tress, there are two approaches:
   * Record the class that each bootstrapped data set predicts and provide an overall prediction to the most commonly occurring one (majority vote).
   * If our classifier produces probability estimates we can just average the probabilities and then predict to the class with the highest probability.
   * Bayes error rate is the lowest possible error rate for a given class of classifier. (can only be estimated)
  
7. Out-of-Bag Error Estimation
  * Since bootstrapping involves random selection of subsets of observations to build a training data set, the remaining non-selected part could be the testing data.
  * On average, each bagged tree makes use of around 2/3 of the observations, so we end up having 1/3 of the observations used for testing.
  * The remaining 1/3 of the observations are referred to as the **out-of- bag (OOB) observations**.
  * The estimated test error using the OOB observations is called the **OOB error**.
  * OOB is the average of error rates from `Tree 1`, `Tree 2` ....`Tree 500`.

  Why is OOB important?
  * When the number of trees B is sufficiently large, OOB error is virtually equivalent to LOOCV error. (LOOCV is the best error you can get.)
  * OOB can be found by printing the bag: `print(bag.carseats)`

8. Importance
  * Bagging improves prediction accuracy at the expense of interpretability
  * **Relative influence plots**: help us decide which variables are most useful in the prediction
    + These plots gives a score for each variable
    + These scores represents the **decrease in MSE** when splitting on a particular variable
      + If the score is 0, then this variable is not important and could be dropped
      + The larger the score, the more important the variable is.

9. Random Forest
  * Random Forest is especially good when there is a **small amount of observations (n)**, but **large amount of features/columns(p)**
  * Build a number of decision trees on bootstrapped training sample, but when building these trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors (Usually m = √p (square root of p))

10. Comparison
  * Decision tree: Validation set approach
    + Use only half of the training set to build a model
    + High variance
    + Decision tree is a special case of random forest
    + Validation set approach is a special case of K-fold CV
  *  Bagging: LOOCV
    + A way to utilise almost all observations in the data set to train a model
    + Bagging is a special case of Random Forest
    + LOOCV is a special case of K-fold CV
  * Random Forest: K-fold CV
    + De-correlate the training sets
    + Produce more reduction on variance

11. Boosting
  * In Bagging, any element has the same probability to appear in a new data set.
  * For Boosting the observations are weighted and therefore some of them will take part in the new sets more often.
  * For overfitting problems, use bagging, because boosting increases overfitting.
  * For bias problems, use boosting, because bagging doesn't reduce bias.

## Week 8. SVM (Support Vector Machine Classifier)

1. Maximal Margin Line Classifier
  * **Margin** is the minimal (perpendicular) distance from all the observations to the separation line
  * Maximal margin line: the line for which the margin is largest
  * Find the minimal distance, and then find the line that has the largest distance
  * **Support vectors** are vectors on the margin, and they support the maximum margin line (support means that if these vectors move, then the max margin line will move as well.)
  * The maximal margin classifier depends only on support vectors
  * no points should be within the margin


2. More than two predictors:
  * Two predictors: a line
  * Three predictors: a plane
  * More than three predictors: a **hyper-plane**

3. Why Maximal Margin Classifiers Are Not sufficient by themselves?

  * Maximal margin hyperplanes may not exist.➔linearly inseparable classes
  * Even if maximal margin hyperplanes exist, they are extremely sensitive to a change in a single observation.→easy to overfit

4. Support Vector Classifier (SVC)
  * Use SVC for linear classification (for non-linear classification, use SVM)

  * SVCs are based on a hyperplane that does not perfectly separate the two classes, in the interest of:
    + Greater robustness to individual observations, and
    + Better classification of most of the training observations.
    + At the cost of worse classification of a few training observations. (the sacrifice)
  * **Soft margin**: We allow some observations to be on the incorrect side of the margin, or even the incorrect side of the hyperplane.
    + points are now allowed in the margin

  * The SVC depends only on support vectors

  * **Cost**: a tuning parameter and is generally chosen via cross validation to specify the cost of validation to the margin. 

    + Cost determines to degree to which the model underfits or overfits the data.
    + When cost is larger, then margin will narrow, and there will be fewer support vectors involved in determining the hyperplane. 
    + Low cost -> wider margin ->low variance -> less likely to overfit
    + When cost reduces, then margin 
    + Classifier highly fit to the data
    + Low bias, high variance
    + In the book, budget is the opposite of the cost. The higher the budgest, the smaller the cost.

  * Which points should influence optimality?
    + All points
      + Linear regression
      + Naive Bayes
      + Linear discriminant analysis
    + Only "difficult points" close to decision boundary
      + Support vector machines
      + Logistic regression (kind of) [See section 9.5 for more details]

  * Libraries for SVC and SVM
    + `e1071` library
    + `LiblineaR` library (useful for very large linear problems)

  * rnorm:generate a list of numbers where mean is 0 and standard deviation is 1

  * Smaller cost ➔ a larger number of support vectors, a wider margin

  *  tune() in e1071 library performs 10-fold cross-validation

3. Support Vector Machine Classifier (SVM)

  * SVM is used for non-linear classification 

  * SVM maps data into a high-dimensional feature space including non-linear features, then use a linear classifier there

  * **Gamma** in RBF Kernel (Radial Basis Function(Gaussian)):

    + Gamma controls the shapes of the "peaks" where you raise the points in the higher dimensional space
      + smaller gamma: softer, broder bumps
      + larger gamma: pointed bumps 
      + the larger the gamma -> high variance, low bias -> likely to overfit

    + Gamma determines which points can determine the decision boundary, and is used to tune the balance between variance and bias
      + large gamma: the decision boundary is only dependent on the points that are very close to it
        + wiggly, jagged boundary (a lot of weight carried by the nearby points)
        + low bias and high variance -> overfitting
      + small gamma: the decision boundary is dependent even on the points that are far away from it
        + smooth boundary
        + high bias and low variance -> not likely to overfit

  * Change the value of kernel in the svm() function
    + **Polynomial** kernel: `kernel=“polynomial”`
      + Use degree argument to specify a degree for the polynomial kernel
    + **Radial** kernel: `kernel=“radial”`
      + Use **gamma argument** to specify a value of γ for the radial basis kernel
    + if we don't know which one to use, use radial (the whole list of options are: linear SVM, RBF SVM (Radial Basis Function), Poly SVM, Sigmoid SVM)


## Week 9. Clustering (Unsupervised Learning)

1. **Clustering** refers to a set of techniques for finding subgroups, or clusters, in a dataset

2. Use **variance** to measure whether members within a group are similar to each other

3. **K-means clustering**: To perform K-means clustering, one must first specify the desired number of clusters K. The objective is to have a minimal "within-cluster-variation”, i.e. the elements within a cluster should be as similar as possible.

  * We can get the minimal "within-cluster-variation" by minimising the sum of pairwise
squared Euclidean distances.

  * "Within-cluster-variation" is `W(c)=sum(C)/# of C`

  * Local optimums
    + Because the K-means algorithm finds a local rather global optimum, it can
      + get stuck in “local optimums” and
      + not find the best solution
      + The results obtained will depend on the initial (random) assignment of each observation
    + Hence, it is important to
      + run the algorithm multiple times
      + with random starting points to find a good solution (i.e. the one with the lowest within-cluster-variation)

  * "**tot.withinss**": The total within-cluster sum of squares. Our objective is to minimise this.

  * Strongly recommend to use a large value of nstart to avoid local optimum (e.g. nstart=100 or 200)

  * It is important to set a random seed before performing K-means clustering - so that we can repeat the result in the future


4. Hierarchical Clustering

  * K-Means clustering requires choosing the number of clusters K. If we don’t want to do that, an alternative is to use Hierarchical Clustering.

  * Two types of hierarchical clustering: Agglomerative and Divisive

  * **Agglomerative** (i.e. Bottom-Up)
    + Start with all points in their own group
    + Until there is only one cluster, repeatedly: merge the two groups that have the smallest dissimilarity
    + this method is easier than divisive. 
    + the disadvantage is that merging is **not reversible** 

  * **Divisive** (i.e., top-down)
    + Start with all points in one cluster
    + Until all points are in their own cluster, repeatedly: split the group into two resulting in the biggest dissimilarity
    + the advantage is that you may be able to stop early
    + the spliting is linear, while the agglomerative is quadruple 

  * **Hierarchical Agglomerative Clustering (HAC) Algorithm**
    + Start with each point as a separate cluster (n clusters)
    + Calculate a measure of dissimilarity between all points/clusters
    + Fuse two clusters that are most similar so that there are now n-1 clusters
    + Fuse next two most similar clusters so there are now n-2 clusters
    + Continue until there is only 1 cluster

  * How to define dissimilarity between clusters?
    There are 4 options:
    + Single Linkage
      + **Single linkage score** is the distance of the closest pair (closest neighbour)
      
      + Cut interpretation: Cut interpretation: for each observation obsi, there is another observation obsj in its cluster and their distance is <=0.9

      + Shortcomings: Single linkage suffers from **chaining**. In order to merge two groups, only need one pair of points to be close, irrespective of all others. Therefore, clusters can be too spread out, and not as compact as the closest pair are to each other.

    + Complete Linkage
      + **Complete linkage score** is the distance of the furthest pair (furthest neighbour)
      + Cut interpretation: for each observation obsi, every observation obsj in its cluster satisfies that their distance is <=5
      + Shortcomings: Complete linkage avoids chaining, but suffers from **crowding**. Because its score is based on the worst-case dissimilarity between pairs, a point can be closer to points in other clusters than to points in its own cluster. Clusters are compact, but not far apart enough.
        + example: guess which music you'd like to hear

    + Average Linkage
      + **Average linkage score** is the average distance across all pairs
      + Shortcomings: It is not clear what properties the resulting clusters have when we cut an average linkage tree at given height h. Single and complete linkage trees each had simple interpretations.

    + Centroid Linkage
      + **Centroid linkage score** is the distance between the cluster centroids (cluster average)
      + Shortcomings: Centroid linkage may produce a dendrogram with inversions – mess up the visualisation, hard to interpret
      + Inversion: The height of a parent node is lower than the height of its children

    + Single, complete, average linkage are widely used by statisticians. (average and complete are preferred over single)
      + Average and complete linkage are generally preferred over single linkage, as they tend to yield evenly sized clusters
      + Single linkage tends to yield extended clusters to which single leaves are
fused one by one

    + Centroid linkage is often used in biology.
      + It is simple, easy to understand and easy to implement
      + But it may have inversion (i.e. parent node being lower than child node)

5. `dist()` function for Hierarchical Clustering
  * `dist()` computes the distances between the rows of a data matrix. If a matrix A has n rows, then dist(A) is of size `(n-1)×(n-1)`.

6. `cutree()` function
  * To determine the cluster labels for each observation associated with a given cut of the dendrogram, we can use the cutree() function

7. Choice of Dissimilarity Measure
  * Euclidean distance vs. correlation based distance

  * Using **Euclidean distance**, customers who have purchases very little will be clustered together

  * Using **correlation measure**, customers who tend to purchase the same types of products will be clustered together even if the magnitude of their purchase may be quite different

8. Need to standardise the Variables
  * If the variables are measured on different scales, we might want to scale the variables to have standard deviation one
  
  * In this way each variable will in effect be given equal importance in the hierarchical/K-means clustering performed

  * This actually applies to many statistical learning methods as a pre- processing step.
  * in r, use `scale(x)` to standarise the variables


## Week 10. Dimension Reduction (Principal Component Analysis) - Unsupervised approach

1. PCA can be used for reducing dimensionality (the number of features), and for data visualisation

2. Given a set of points, how do we know if they can be compressed with PCA?
  * By looking at the **covariance**(the relationship between two variables) between points.

3. 
  * The **first** principal component is the best straight line you can fit to the **data**.
  * The **second** principal component is the best straight line you can fit to the **errors** from the first principal component. Second PC is perpendicular to the first PC. 
  * The **third** principal component is the best straight line you can fit to the errors from the first and second principal components, etc. Third PC is perpendicular to the first and second PC. 

4. PCA: can only find a linear relationship of the factors we have. For a non-linear relationship, we should use log transformation.

5. Matrix size: `(m × n) × (n × k) = m × k` 

6. Covariance matrix, Eigenvectors and Eigenvalues
  * Given a matrix A, and a vector x, when A is used to transform x, the result is `y = Ax`
  * **Eigenvector**: vectors which does not change its direction under transformation, however allow the vector magnitude to vary by scalar `λ`.
    +  such special x are called **eigenvectors** and λ are called **eigenvalues**
    + eigenvector and eigenvalues are in pairs

7. All the eigenvectors of a symmetric matrix **are perpendicular**, i.e., at right angles to each other, no matter how many dimensions there are.
  * In maths terminologies, another word for perpendicular is **orthogonal**.
  * Since the length of an eigenvector does not affect its direction, we can scale
(or standardise) eigenvectors so that they have the same length of 1.
  * If we scale the eigenvector, the eigenvalue stays the same
  * Eigenvectors can only be found for square matrices. (i.e vertical and horizontal axis should be of the same degree)
  * For a matrix `Ap×p` , there are **at most p pairs of (eigenvector, eigenvalue)**.

8. How PCA works
  * Given a dataset DS with n observations and p features, we can build a covariance matrix `Cp×p`
  * Compute a set of pairs of (eigenvector, eigenvalue)
    ```r
    (e1, 0.382), (e2, 2.618), (e3, 1.439), ..., (ep, 0.00096)
    ```
  * Sort them by the eigenvalues in a descending order
    ```r
    (e2, 2.618), (e3, 1.439), (e1, 0.382), ...
    ```
  * The first eigenvector is the first principal component (PC), the second eigenvector is the second PC, and so on.
    ```
    The first PC: e2
    The second PC: e3
    The third PC: e1
    ```
  * Takeaway: **The larger the eigenvalue is, the more important the direction/Principle Components is**

9. How to do PCA
  * Step 1: Set new center/origin 
  * Step 2: Calculate the covariance matrix
  * Step 3: Calculate the eigenvectors and eigenvalues of the covariance matrix
  * Step 4: Select the Principle Components (PC) - first PC and second PC are perpendicular to each other

10. Do PCA in R:
  ```r
  > DF <- data.frame(
       Maths=c(80, 90, 95),
       Science=c(85, 85, 80), 
       English=c(60, 70, 40), Music=c(55, 45, 50))
  > pr.marks <- prcomp(DF, scale=FALSE)
  > pr.marks$rotation #omit PC3
           PC1          PC2 
    Maths 0.27795606 0.76772853 
    Science -0.17428077 -0.08162874 
    English -0.94200929 0.19632732 
    Music 0.07060547 -0.60447104
  > biplot(pr.marks, scale=FALSE)
  # biplot will only plot two PCs (i.e. two dimensions)
  ```

11. Scaling
  * We use scaling because variables may have different units, and different variance
  * Calculate variance for each variable:
    ```r
    apply(DF,2,var)
    ```

  * If performing PCA on the unscaled data, then the first PC will place almost all its weighting on English (most variance variable) but little on Science (least variance variable)

  * We usually scale the variables to have **standard deviation one** before we perform PCA.

  * Use scaling in R:
  ```r
  pr.marks.s <- prcomp(DF, scale = TRUE)
  biplot(pr.marks.s, scale=0)
  ```

12. The Proportion of Variance Explained ("PVE", i.e. how much information is lost by doing dimension reduction)
  * Scree plot vs. cumulative PVE

13. Deciding How Many PCs to Use

  * Scree plot: use the elbowing method to determine when to stop using more Principle Components 

  * In practice, we look at first few PCs to find interesting patterns.
    + If no interesting patterns are found, further PCs are unlikely to be interesting
    + If first few PCs are interesting, continue to look at subsequent PCs, until no further interesting patterns are found



## Week 11. Applications
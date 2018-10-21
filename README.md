# Big Data Analytics using R
## Master of Data Science at Birkbeck, University of London
### 10/2018

Summary:

### Syllabus 
1. Basic Statistics + Linear Regression
2. Linear Regression
3. Logistic Regression
4. Cross Validation
5. Decision Trees
6. Ensemble Methods
7. SVM
8. Clustering
9. Dimension Reduction
10. Applications

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
   *Popluation regress line is the best <b>linear</b> approximation to the true relationship between X and Y
3. Least square line: `y(with hat) = 𝛽0(with hat) + 𝛽1(with hat)x`

#### Standard error (SE)
Standard error is the average amount that this estimate 𝜇-hat differs from μ.  

```SE= variance / n```

The more observations we have, the smaller the SE is. 

#### Hypothesis Tests
Is `B1 = 0` or not? If we can't be sure that B1 `!= then` there is no point using X as the predicto. 

1. <b>Null hypothesis</b> - There is no relationship between X and Y
2. <b>Alternative hypothesis</b> - There is some relationship between X and Y
3. The higher <b>t-value</b> is (equivalently p-value is small), the more possible X and Y are related, then we can be sure that B1(hat) is not 0.
  * We can reject the Null Hypothesis.
  * We declare a relationship to exist between X and Y.
  
4. <b>P-value</b>: P values evaluate how well the sample data support that the null hypothesis is true. It measures how compatible your data are with the null hypothesis
  * A smallp-value (typically ≤ 0.05) indicates yoursample provides strong evidence against the null hypothesis, so you reject the null hypothesis.
  * A largep-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
  * Typical p-value cutoffs for rejecting the null hypothesis are 5 or 1%  

5. Summary of t-value and p-value  
* 
* The t-test produces a single value,t, which grows larger as the difference between the means of two samples grows larger;  
* t does not cover a fixed range such as 0 to 1 like probabilities do;
* You can convert a t-value into a probability, called a p-value;  
* The p-value is always between 0 and 1 and it tells you the probability of the difference in your data being due to sampling error;  
* The p-value should be lower than a chosen significance level (0.05 for example) before you can reject your null hypothesis.


#### Step 3: How to assess accuracy of the model
1. Residual Standard Error (RSE)
* It measures the degress to which the model fits the data (e.g. should this be linear vs. non-linear regression?)
*  RSE is the estimate of the standard deviation of `ε`
*  RSE has a unit, and it has the same unit as `Y`
*  Quantifies average amount that the response will deviate from the population regression

2. Measuring the extent to which the model fits the data  
R<sup>2</sup> statistic
* Some of the variation in Y can be explained by variation in the X’s and some cannot.
* R<sup>2</sup> tells you the <b>proportion of variance</b> of Y that can be explained by X.
* <b>The higher the R<sup>2</sup>, the better fit the model is.</b>
* In simple linear regression, R<sup>2</sup> == Cor(X,Y)<sup>2</sup>

3. `Cor(X,Y) = 0` means there is no <b>linear</b> relationship between X and Y, but there could be other relationship.

4. Multiple R-squared and adjusted R-squared:
* Model with multiple variables: use adjusted R-squared 
* Model with single variable: use R squared and adjusted R squared interchangably 
* Generally <b>use adjusted R-squared</b> as preference

5. Predicting mean is a lot more accurate than predicting an individual

6. <b>Prediction interval (PI)</b> is an estimate of an interval in which future observations (particular individuals) will fall, with a certain probability, given what has already been observed.

7. <b>Confidence intervals</b> predict <b>confidence intervals</b> for the <b>mean</b>, <b>prediction intervals</b> for the <b>individuals</b>.

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







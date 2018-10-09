# Big Data Analytics using R
## Master of Data Science at Birkbeck, University of London
### 10/2018

Summary:

Feedback:

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

## Week 3 and onwards: Inferential statistics




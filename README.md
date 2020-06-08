
# ANOVA - Analysis of Variance
Today, we will be learning ANOVA, a generalized form of comparing mean across multiple groups. 
Agenda today:
1. Compare t-tests and ANOVA
2. Learn to calculate ANOVA & details 
3. Implement ANOVA in python
    - using statsmodel


## Part I. T tests or ANOVA?
**ANOVA** or *Analysis Of Variance*  provides a statistical test of whether two or more population means are equal, and therefore generalizes the t-test beyond two means.

Suppose we want to compare whether multiple groups differ in some type of measures. For example, we have collected mood data grouped by four types of weather - sunny, raining, overcast, or cloudy, and we want to find out whether there is a difference in mood across different weather. What tests would you use?

A natural reaction would be to conduct multiple t-tests. However, that comes with many drawbacks. First, you would need $\frac{n(n-1)}{2}$ t tests, which come out to 6 tests. Having more tests meaning having higher chance of making type I error. In this case, our original probability of making type I error grew from 5% to 5% x 6 = 30%! By conducting 6 tests and comparing their mean to each other, we are running a huge risk of making false positives. How then, can we combat this? -- ANOVA!

Instead of looking at each individual difference, ANOVA examines the ratio of variance between groups, and variance within groups, to find out whether the ratio is big enough to be statistically significant. 

#### T Test statistics 
##### One sample
$t = \frac{x\bar - \mu}{\frac{s}{\sqrt n}}$

##### Two sample
$$ t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{s^2 \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}}$$

where $s^2$ is the pooled sample variance,

$$ s^2 = \frac{\sum_{i=1}^{n_1} \left(x_i - \bar{x_1}\right)^2 + \sum_{j=1}^{n_2} \left(x_j - \bar{x_2}\right)^2 }{n_1 + n_2 - 2} $$

We can also say that t test is a special case of ANOVA in that we are comparing the means of only two groups.


#### ANOVA - the F test
$F = \frac{MS_{bet}}{MS_{within}}$


```python
one = np.random.normal(0,3,100)
two = np.random.normal(1,3,100)
```


```python
from scipy import stats
stats.f_oneway(one, two)
```




    F_onewayResult(statistic=11.219321475072164, pvalue=0.0009691569451370045)




```python
# Identical p_values
from scipy.stats import ttest_ind
t = ttest_ind(one, two, equal_var=True)
t

```




    Ttest_indResult(statistic=-3.349525559698292, pvalue=0.0009691569451369975)




```python
# Two-sample t-stat equals F-stat squared
t.statistic**2
```




    11.219321475072157



<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/image046.png" width="500">

#### Between Group Variability

Measures how much the means of each group vary from the mean of the overall population



<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/image0171.png" width="500">
    

#### Within Group Variability

Refers to variations caused by differences within individual groups.  

Consider the given distributions of three samples below. As the spread (variability) of each sample is increased, their distributions overlap and they become part of a big population.
<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/image031.png" width="500">

Now consider another distribution of the same three samples but with less variability. Although the means of samples are similar to the samples in the above image, they seem to belong to different populations.

<img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/image033.png" width="500">


## Trios: 
Take 2 minutes in groups of 3 to discuss which of these following trios would have high f-stats, and which would have low.

Plot them if, if that would help.

Run the f_oneway funciton scypy.stats to check your conclusions


```python
import seaborn as sns
# Create three sets of data without much difference in means
np.random.seed(42)

a = np.random.normal(20,20,20)
b = np.random.normal(22,20,20)
c = np.random.normal(19,20,20)

one = np.random.normal(20,1,20)
two = np.random.normal(22,2,20)
three = np.random.normal(19,3,20)

x = np.random.normal(20,1,20)
y = np.random.normal(22,200,2)
z = np.random.normal(19,3,30)

four = np.random.normal(20,10,20)
five = np.random.normal(20,10,20)
six = np.random.normal(30,10,20)


```


```python
#__SOLUTION__
fig, ax = plt.subplots(2,2, figsize=(15,10))

sns.distplot(a, hist=False, ax=ax[0][0])
sns.distplot(b, hist=False, ax=ax[0][0])
sns.distplot(c, hist=False, ax=ax[0][0])

sns.distplot(one, hist=False, ax=ax[0][1])
sns.distplot(two, hist=False, ax=ax[0][1])
sns.distplot(three, hist=False, ax=ax[0][1])

sns.distplot(x, hist=False, ax=ax[1][0])
sns.distplot(y, hist=False, ax=ax[1][0])
sns.distplot(z, hist=False, ax=ax[1][0])

sns.distplot(four, hist=False, ax=ax[1][1])
sns.distplot(five, hist=False, ax=ax[1][1])
sns.distplot(six, hist=False, ax=ax[1][1])

# Large variance overcomes the difference in means, so low fstat
print(stats.f_oneway(a,b,c))
# Difference in means is able to overcome the small variance, so high fstat
print(stats.f_oneway(one,two,three))
# Difference in means is able to overcome variance, since the small sample size of large variance 
# is outweighed by the small variance in larger sample sizes
print(stats.f_oneway(x,y,z))
# Two samples are indistinguishable, but the third sample's large difference in means are able to overcome
# the within group variance to distinguish itself.
print(stats.f_oneway(four,five,six))
```

    F_onewayResult(statistic=0.06693195000987277, pvalue=0.9353322377145488)
    F_onewayResult(statistic=10.020229924359118, pvalue=0.00018661463272077867)
    F_onewayResult(statistic=48.70686089104815, pvalue=2.254500528606684e-12)
    F_onewayResult(statistic=3.424034327506795, pvalue=0.03942049651729238)



![png](index_files/index_15_1.png)



```python
print(stats.f_oneway(a,b,c))
print(stats.f_oneway(one,two,three))
print(stats.f_oneway(x,y,z))
print(stats.f_oneway(four,five,six))
```

    F_onewayResult(statistic=0.06693195000987277, pvalue=0.9353322377145488)
    F_onewayResult(statistic=10.020229924359118, pvalue=0.00018661463272077867)
    F_onewayResult(statistic=48.70686089104815, pvalue=2.254500528606684e-12)
    F_onewayResult(statistic=3.424034327506795, pvalue=0.03942049651729238)


## Part II. Calculating ANOVA 
In this section, we will learn how to calculate ANOVA without using any pacakges. All we need to calculate is:

Total Sum of Squares is the every value minus the mean of the entire population, or in other words, the total varaince of the population. 
- $SS_t$ = $\sum (X_{ij} - \bar X)^2$

The total sum of squares can be broken down into the sum of squares between and the sum of squares within.
- $SS_t =  SS_b+SS_w  $

The sum of squares between accounts for variance in the dataset that comes from the difference between the mean of each sample, divided by the mean of means. That is, the sum of the difference between each group mean and the mean of means for each data point: 
- $SS_b$ = $\sum(n_i(\bar X - \bar X_i)^2) $

The sum of squares within accounts for variance that comes from within each sample.  That is, the sum of the variances of each group weighted by the group's degrees of freedom:
- $SS_w$ = $\sum (n_i - 1) s_i ^ 2$  


- $MS_b$ = $\frac{SS_b}{DF_b}$
- $MS_w$ = $\frac{SS_w}{DF_w}$


- $F$ = $\frac{MS_b}{MS_w}$

Degrees of Freedom for ANOVA:
-  $DF_{between}$ = k - 1
- $DF_{within}$ = N - k
- $DF_{total}$ = N - 1

Notations:
- k is the number of groups
- N is the total number of observations
- n is the number of observations in each group

Like regression and t-test, we can also perform hypothesis testing with ANOVA. 

- $H_0$ : $\mu{_1}$ = $\mu_2$ = $\mu_3$ = $\mu_4$
- $H_a$ : $H_0$ is not true

Under the null hypothesis (and with certain assumptions), both quantities estimate the variance of the random error, and thus the ratio should be small. If the ratio is large, then we have evidence against the null, and hence, we would reject the null hypothesis.


```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
df = pd.read_csv('bikeshare_day.csv')
df.head()
# cnt is the outcome we are trying to predict
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.344167</td>
      <td>0.363625</td>
      <td>0.805833</td>
      <td>0.160446</td>
      <td>331</td>
      <td>654</td>
      <td>985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.363478</td>
      <td>0.353739</td>
      <td>0.696087</td>
      <td>0.248539</td>
      <td>131</td>
      <td>670</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.196364</td>
      <td>0.189405</td>
      <td>0.437273</td>
      <td>0.248309</td>
      <td>120</td>
      <td>1229</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.200000</td>
      <td>0.212122</td>
      <td>0.590435</td>
      <td>0.160296</td>
      <td>108</td>
      <td>1454</td>
      <td>1562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-05</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.226957</td>
      <td>0.229270</td>
      <td>0.436957</td>
      <td>0.186900</td>
      <td>82</td>
      <td>1518</td>
      <td>1600</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we need to conduct a little bit feature engineering to encode 
df['season_cat'] = df.season.apply(lambda x: 'spring' if x == 1 else 
                                           (
                                            'summer' if x == 2 else (
                                                'fall' if x == 3 else 'winter')
                                           )
                                      )
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
      <th>season_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.344167</td>
      <td>0.363625</td>
      <td>0.805833</td>
      <td>0.160446</td>
      <td>331</td>
      <td>654</td>
      <td>985</td>
      <td>spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.363478</td>
      <td>0.353739</td>
      <td>0.696087</td>
      <td>0.248539</td>
      <td>131</td>
      <td>670</td>
      <td>801</td>
      <td>spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.196364</td>
      <td>0.189405</td>
      <td>0.437273</td>
      <td>0.248309</td>
      <td>120</td>
      <td>1229</td>
      <td>1349</td>
      <td>spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.200000</td>
      <td>0.212122</td>
      <td>0.590435</td>
      <td>0.160296</td>
      <td>108</td>
      <td>1454</td>
      <td>1562</td>
      <td>spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-05</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.226957</td>
      <td>0.229270</td>
      <td>0.436957</td>
      <td>0.186900</td>
      <td>82</td>
      <td>1518</td>
      <td>1600</td>
      <td>spring</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create a boxplot
# 1 is spring, 2 is summer, 3 is fall, and 4 is winter
df.boxplot('cnt', by='season_cat', figsize=(6,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2d9f5320>




![png](index_files/index_23_1.png)



```python
from scipy import stats
# We could perform two sample t-tests for each sample.

# subset the dataframe  by season and isolate the dependent variable
spring = df[df.season_cat == 'spring'].cnt
fall = df[df.season_cat == 'fall'].cnt
summer = df[df.season_cat == 'summer'].cnt
winter = df[df.season_cat == 'winter'].cnt

# We could run independent t-tests for each combination
# But that increases the chance of making a type I (False Positive) error
# Also, as your groups increase, the number of tests may become infeasable
print(stats.ttest_ind(spring, fall))
print(stats.ttest_ind(spring, summer))
print(stats.ttest_ind(spring, winter ))
print(stats.ttest_ind(fall, summer))
print(stats.ttest_ind(fall, winter))
print(stats.ttest_ind(summer, winter))
```

    Ttest_indResult(statistic=-20.40505135948835, pvalue=2.134072968524431e-62)
    Ttest_indResult(statistic=-14.65873026929708, pvalue=1.5284822271363832e-38)
    Ttest_indResult(statistic=-12.933694332032188, pvalue=1.2022067175230552e-31)
    Ttest_indResult(statistic=3.9765418611661243, pvalue=8.411509811510022e-05)
    Ttest_indResult(statistic=5.541003097872063, pvalue=5.7789091515026665e-08)
    Ttest_indResult(statistic=1.480020595990678, pvalue=0.13974231789501412)



```python
# Code for me:

mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian', ]

new_choice = np.random.choice(mccalister)
print(new_choice)
mccalister.remove(new_choice)
```

    Karim



```python
# Instead, we can run an ANOVA test to see if there is statistically significant differences between the means.

# Let's code the f-stat together
# Here is the pseudo code

# Define k 

# Define N

# Calculate SSB

# Calculate SSW

# Calculate DFw

# Calculate DFb

# Calculate MSb

# Calculate MSw

# Calculate F-stat


```


```python
#__SOLUTION__
# Instead, we can run an ANOVA test to see if there is statistically significant differences between the means.

# Let's code the f-stat together

# Define k 
k = 4
# Define N

N = len(df)
# Here is the pseudo code

# Calculate SSb
# This is the amount of variability accounted for by the difference in means
mean_of_means = df.cnt.mean()

ssbs = 0

for season in df.season_cat.unique():
    
    group_mean = df[df.season_cat == season].cnt.mean()
    sum_of_s = (group_mean - mean_of_means)**2
    ssbs += len(df[df.season_cat == season])* sum_of_s
    
ssb = ssbs
# Calculate SSw

ssw = 0
for season in df.season_cat.unique():
    n = len(df[df.season_cat == season])
    group_var = np.var(df[df.season_cat == season].cnt, ddof=1)
    ssw += group_var*(n-1)

# Calculate DFw
DFw = N-k

# Calculate DFb
DFb = k-1
# Calculate MSb

MSb = ssb/DFb
# Calculate MSw
MSw = ssw/DFw
# Calculate F-stat

f_stat = MSb/MSw
f_stat
```




    128.7696215657079



### 2.2 Calculate ANOVA using statsmodel


```python
data.boxplot('cnt', by = 'season_cat')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a269bc940>




![png](index_files/index_29_1.png)



```python
from statsmodels.formula.api import ols
import statsmodels.api as sm
# why can we use ols in this case?
anova_season = ols('cnt~season_cat',data=data).fit()
# anova_season.summary()
```


```python
# examine the anova table 
anova_table = sm.stats.anova_lm(anova_season, type=2)
print(anova_table)
```

                   df        sum_sq       mean_sq           F        PR(>F)
    season_cat    3.0  9.505959e+08  3.168653e+08  128.769622  6.720391e-67
    Residual    727.0  1.788940e+09  2.460715e+06         NaN           NaN


<img src="attachment:Screen%20Shot%202019-06-03%20at%2010.36.09%20AM.png" width="400">


```python
data.columns
```




    Index(['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
           'casual', 'registered', 'cnt', 'season_cat'],
          dtype='object')



## Perform an ANOVA with scipy


```python
stats.f_oneway(data['cnt'][data['season_cat'] == 'summer'],
                data['cnt'][data['season_cat'] == 'fall'], 
                data['cnt'][data['season_cat'] == 'winter'],
                data['cnt'][data['season_cat'] == 'spring'])
```




    F_onewayResult(statistic=128.76962156570784, pvalue=6.720391362913176e-67)



#### Next steps
Just because we have rejected the null hypothesis, it doesn't mean we have conclusively showed which group is significantly different from which - remember, the alternative hypothesis is "the null is not true". 

We need to conduct post hoc tests for multiple comparison to find out which groups are different, the most prominent post hoc tests are:
- LSD (Least significant difference)
    - $t\sqrt \frac{MSE}{n^2}$
- Tukey's HSD 
    - $q\sqrt \frac{MSE}{n}$
    
https://www.statisticshowto.com/studentized-range-distribution/#qtable
    
After calculating a value for LSD or HSD, we compare each pair wise mean difference with the LSD or HSD difference. If the pairwise mean difference exceeds the LSD/HSD, then they are significantly different.


```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(df.cnt, df.season_cat))
```

        Multiple Comparison of Means - Tukey HSD, FWER=0.05     
    ============================================================
    group1 group2  meandiff  p-adj    lower      upper    reject
    ------------------------------------------------------------
      fall spring -3040.1706  0.001 -3460.8063 -2619.5349   True
      fall summer  -651.9717  0.001 -1070.8566  -233.0867   True
      fall winter  -916.1403  0.001 -1338.5781  -493.7025   True
    spring summer  2388.1989  0.001  1965.3265  2811.0714   True
    spring winter  2124.0303  0.001  1697.6383  2550.4224   True
    summer winter  -264.1686 0.3792  -688.8337   160.4965  False
    ------------------------------------------------------------


## Two-Way ANOVA:
Using one-way ANOVA, we found out that the season was impactful on the mood of different people. What if the season was to affect different groups of people differently? Say maybe older people were affected more by the seasons than younger people.

Moreover, how can we be sure as to which factor(s) is affecting the mood more? Maybe the age group is a more dominant factor responsible for a person's mode than the season.

For such cases, when the outcome or dependent variable (in our case the test scores) is affected by two independent variables/factors we use a slightly modified technique called two-way ANOVA.

### Resources

https://www.analyticsvidhya.com/blog/2018/01/anova-analysis-of-variance/
    
https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/anova/how-to/one-way-anova/before-you-start/overview/

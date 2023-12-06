---
layout: default
title: "Introduction by Examples"
---
# Introduction by Examples
We introduce ``fasttrees`` through self-contained examples.


# Credit approval data set
Let's walk through an example of using the Fast-and-Frugal Tree classifier to predict whether we should approve a request for a credit card.
The data set is from the [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval) hosteb by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
Each row in the feature matrix `X` hence represents historical attributes we collected within the credict card approval process, and the information whether the request was approved or denied.
We have 15 features and 1 target for 690 credit card request.
The data types of the features are either categorical, integer, or real.


First, we import the Fast-and-Frugal Tree classifier.

```python
from fasttrees import FastFrugalTreeClassifier
```

Now let’s get some data to fit our classifier to. Fast-and-Frugal Trees tend to do well on real-world data prone to (human) error, as they disregard information that doesn’t seem very predictive of the outcome. A typical use case is in an operational setting where humans quickly have to take decisions. You could then fit a fast-and-frugal tree to the data in advance, and use the simple resulting tree to quickly make decisions.

Let’s load the data as CSV to a Pandas dataframe, using the `ucimlrepo` package.

```python
# code taken from https://archive.ics.uci.edu/dataset/27/credit+approval
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
credit_approval = fetch_ucirepo(id=27) 
  
# data (as pandas dataframes) 
X = credit_approval.data.features 
y = credit_approval.data.targets 
```

```python
# metadata 
print(credit_approval.metadata) 
  
# variable information 
print(credit_approval.variables) 
```

`X` is the feature matrix, and `y` the target we want to predict, in other words the credit decision. This credict decision is denoted by a `+` for approved and a `-` for declined. For our classifier to work, we need to convert this to a boolean.

```python
y.A16 = y.A16.apply(lambda x: True if x=='+' else False).astype(bool)
```


Your feature matrix `X` and your target `y` should now look similar to the following.
```python
X.head()
```
<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>A15</th>\n      <th>A14</th>\n      <th>A13</th>\n      <th>A12</th>\n      <th>A11</th>\n      <th>A10</th>\n      <th>A9</th>\n      <th>A8</th>\n      <th>A7</th>\n      <th>A6</th>\n      <th>A5</th>\n      <th>A4</th>\n      <th>A3</th>\n      <th>A2</th>\n      <th>A1</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>202.0</td>\n      <td>g</td>\n      <td>f</td>\n      <td>1</td>\n      <td>t</td>\n      <td>t</td>\n      <td>1.25</td>\n      <td>v</td>\n      <td>w</td>\n      <td>g</td>\n      <td>u</td>\n      <td>0.000</td>\n      <td>30.83</td>\n      <td>b</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>560</td>\n      <td>43.0</td>\n      <td>g</td>\n      <td>f</td>\n      <td>6</td>\n      <td>t</td>\n      <td>t</td>\n      <td>3.04</td>\n      <td>h</td>\n      <td>q</td>\n      <td>g</td>\n      <td>u</td>\n      <td>4.460</td>\n      <td>58.67</td>\n      <td>a</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>824</td>\n      <td>280.0</td>\n      <td>g</td>\n      <td>f</td>\n      <td>0</td>\n      <td>f</td>\n      <td>t</td>\n      <td>1.50</td>\n      <td>h</td>\n      <td>q</td>\n      <td>g</td>\n      <td>u</td>\n      <td>0.500</td>\n      <td>24.50</td>\n      <td>a</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>3</td>\n      <td>100.0</td>\n      <td>g</td>\n      <td>t</td>\n      <td>5</td>\n      <td>t</td>\n      <td>t</td>\n      <td>3.75</td>\n      <td>v</td>\n      <td>w</td>\n      <td>g</td>\n      <td>u</td>\n      <td>1.540</td>\n      <td>27.83</td>\n      <td>b</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0</td>\n      <td>120.0</td>\n      <td>s</td>\n      <td>f</td>\n      <td>0</td>\n      <td>f</td>\n      <td>t</td>\n      <td>1.71</td>\n      <td>v</td>\n      <td>w</td>\n      <td>g</td>\n      <td>u</td>\n      <td>5.625</td>\n      <td>20.17</td>\n      <td>b</td>\n    </tr>\n  </tbody>\n</table>

```python
y.head()
```
<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>A16</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>+</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>+</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>+</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>+</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>+</td>\n    </tr>\n  </tbody>\n</table>


After preprosssing our data set, we can now split the data set into a training and test set.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
```

Finally, we can instantiate our fast-and-frugal tree classifier and fit it. The fit can take a few seconds.
```python
fc = FastFrugalTreeClassifier()
fc.fit(X_train, y_train)
```

We can take a look at the resulting tree, which can be used for decision making.
```python
fc.get_tree()
```
```
	IF NO 	feature 	direction 	threshold 	IF YES
0 	decide NO 	A8 	in 	(‘t’,) 	↓
1 	↓ 	A10 	> 	1 	decide YES
2 	↓ 	A9 	in 	(‘t’,) 	decide YES
3 	decide NO 	A7 	> 	1.25 	decide YES
```

Now somebody making a decision can simply look at the 3 central columns, which read, for example A8 in ('t',) and have a look whether this is the case. If it isn’t, they take the action in IF NO, which would be to decide NO (in this case that would mean not to grant this specific person a credit). If it is, they take the action in IF YES, which is to look at the next feature, for which they then repeat the process.

How well does this simple tree classifier perform? Let’s score it against the test data. By default the balanced accuracy score is used:

```python
fc.score(X_test, y_test)
```

This returns a balanced accuracy of 0.86, pretty good!

We can also have a look at how much information it actually used to make its decisions:

```python
fc.get_tree(decision_view=False)
```

```
	feature 	direction 	threshold 	type 	balanced_accuracy_score 	fraction_used 	exit
0 	A8 	in 	(‘t’,) 	categorical 	0.852438 	1 	0
1 	A10 	> 	1 	numerical 	0.852438 	0.528139 	1
2 	A9 	in 	(‘t’,) 	categorical 	0.852438 	0.238095 	1
3 	A7 	> 	1.25 	numerical 	0.852438 	0.192641 	0.5
```

While the first cue is used for all decisions, the second is only used for 52% of all decisions. This means that 48% of decisions could be made by just looking at one single feature.

Hence, fast and frugal trees provide a very easy way to generate simple decision criteria from a large dataset, which often perform better than more advanced machine learning algorithms, and are much more transparent.

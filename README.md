[![Downloads](https://static.pepy.tech/badge/fasttrees)](https://pepy.tech/project/fasttrees)
![PyPI - Version](https://img.shields.io/pypi/v/fasttrees)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fasttrees)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
![pyling: workflow](https://github.com/dominiczy/fasttrees/actions/workflows/pylint.yml/badge.svg)
[![Upload Python Package](https://github.com/dominiczy/fasttrees/actions/workflows/python-publish.yml/badge.svg)](https://github.com/dominiczy/fasttrees/actions/workflows/python-publish.yml)


# fasttrees
A fast-and-frugal-tree classifier based on Python's scikit learn.

Fast-and-frugal trees are classification trees that are especially useful for making decisions under uncertainty. 
Due their simplicity and transparency they are very robust against noise and errors in data.
They are one of the heuristics proposed by Gerd Gigerenzer in [Fast and Frugal Heuristics in Medical Decision](library.mpib-berlin.mpg.de/ft/gg/GG_Fast_2005.pdf). This particular implementation is based on on the R package [FFTrees](https://cran.r-project.org/web/packages/FFTrees/index.html), developed by Phillips, Neth, Woike and Grassmaier.

## Install
You can install fasttrees using
```
pip install fasttrees
```

## Quick first start

Below we provide a qick first start example with fast-and-frugal trees. We use the popular [iris flower data set (also known as the Fisher's Iris data set)](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x), split it into a train and test data set, and fit a fast-and-frugal tree classifier on the training data set. Finally, we get the score on the test data set.

```python
from sklearn import datasets, model_selection

from fasttrees.fasttrees import FastFrugalTreeClassifier


# Load data set
iris_dict = datasets.load_iris(as_frame=True)

# Split into train and test data set
X_iris, y_iris = iris_dict['data'], iris_dict['target']
X_train_iris, y_train_iris, X_test_iris, y_test_iris = model_selection.train_test_split(
    X_iris, y_iris, test_size=0.4, random_state=42)

# Fit and test fitted tree
fftc = FastFrugalTreeClassifier()
fftc.fit(X_train_iris, y_train_iris)
fftc.score(X_test_iris, y_test_iris)
```

## Usage
Instantiate a fast-and-frugal tree classifier:
```python
fc = FastFrugalTreeClassifier()
```

Fit on your data:
```python
fc.fit(X_train, y_train)
```

View the fitted tree (this is especially useful if the 'predictions' will be carried out by humans in practice):
```python
fc.get_tree()
```

Predict:
```python
preds = fc.predict(X_test)
```

Score:
```python
fc.score(X_test, y_test)
```

## Example
Let's walk through an example of using the Fast-and-Frugal Tree classifier.
First, we import the Fast-and-Frugal Tree classifier.

```python
from fasttrees.fasttrees import FastFrugalTreeClassifier
```

Now let’s get some data to fit our classifier to. Fast-and-Frugal Trees tend to do well on real-world data prone to (human) error, as they disregard information that doesn’t seem very predictive of the outcome. A typical use case is in an operational setting where humans quickly have to take decisions. You could then fit a fast-and-frugal tree to the data in advance, and use the simple resulting tree to quickly make decisions.

As an example of this, let’s have a look at credit decisions. UCI provides a credit approval dataset. Download the crx.data file from the data folder.

Let’s load the data as CSV to a Pandas dataframe:

```python
import pandas as pd
data = pd.read_csv('crx.data', header=None)
```

As there is no header, the columns are simply numbered 1, 2, 3 etc. Let’s make clear they’re attributes by naming them A1, A2, A3 etc.

```python
data.columns = ['A{}'.format(nr) for nr in data.columns]
```

The fasttrees implementation of fast-and-frugal trees can only work with categorical and numerical columns, so let’s assign the appropriate dtype to each column:

```python
import numpy as np

cat_columns = ['A0', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
nr_columns = ['A1', 'A2', 'A7', 'A10', 'A13', 'A14']

for col in cat_columns:
    data[col] = data[col].astype('category')

for col in nr_columns:
    # only recast columns that have not been correctly inferred
    if data[col].dtype != 'float' and data[col].dtype != 'int':
        # change the '?' placeholder to a nan
        data.loc[data[col] == '?', col] = np.nan
        data[col] = data[col].astype('float')
```

The last column is the variable we want to predict, the credit decision. It’s denoted by + or -. For our FastFrugalTreeClassifier to work we need to convert this to boolean:

```python
data['A15'] = data['A15'].apply(lambda x: True if x=='+' else False).astype(bool)
```

Your data should now look something like this:

```
	A0 	A1 	A2 	A3 	A4 	A5 	A6 	A7 	A8 	A9 	A10 	A11 	A12 	A13 	A14 	A15
0 	b 	30.83 	0 	u 	g 	w 	v 	1.25 	t 	t 	1 	f 	g 	202 	0 	True
1 	a 	58.67 	4.46 	u 	g 	q 	h 	3.04 	t 	t 	6 	f 	g 	43 	560 	True
2 	a 	24.5 	0.5 	u 	g 	q 	h 	1.5 	t 	f 	0 	f 	g 	280 	824 	True
3 	b 	27.83 	1.54 	u 	g 	w 	v 	3.75 	t 	t 	5 	t 	g 	100 	3 	True
4 	b 	20.17 	5.625 	u 	g 	w 	v 	1.71 	t 	f 	0 	f 	s 	120 	0 	True
```

Now let’s do a train test split (we use two thirds of the data to train on):

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='A15'), data['A15'], test_size=0.33, random_state=0)

We can now finally instantiate our fast-and-frugal tree classifier. Let’s use the default parameters:

fc = FastFrugalTreeClassifier()
```

Let’s fit the classifier to our training data (this can take a few seconds):

```python
fc.fit(X_train, y_train)
```

We can take a look at the resulting tree, which can be used for decision making:

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

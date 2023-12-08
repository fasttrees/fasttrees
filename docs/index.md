---
layout: default
title: "Home | fasttrees"
nav_order: 1
description: fasttrees is a Python implementation of fast-and-frugal trees, which are a specialisation of decistion trees for binary-classification.
permalink: /
---
# Welcome to fasttrees

`fasttrees` is a Python implementation of fast-and-frugal trees, which are a specialisation of decistion trees for binary-classification.
{: .fs-6 .fw-300 }

[Install fasttrees](#install-fasttrees){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View fasttrees on GitHub](https://github.com/fasttrees/fasttrees){: .btn .fs-5 .mb-4 .mb-md-0 }
[Open an Issue on GitHub](https://github.com/fasttrees/fasttrees/issues){: .btn .fs-5 .mb-4 .mb-md-0 }


# Install fasttrees

You can easily install `fasttrees` via `pip`.

```bash
pip install fasttrees
```

# When to use fast-and-frugal trees

Fast-and-Frugal Trees tend to do well on real-world data prone to (human) error, as they disregard information that doesn’t seem very predictive of the outcome. A typical use case is in an operational setting where humans quickly have to take decisions. You could then fit a fast-and-frugal tree to the data in advance, and use the simple resulting tree to quickly make decisions.

# Usage

```python
from fasttrees import FastFrugalTreeClassifier


# Instantiate a fast-and-frugal tree classifier
fc = FastFrugalTreeClassifier()
# Fit on your data
fc.fit(X_train, y_train)
# View the fitted tree (this is especially useful if the 'predictions' will be carried out by humans in practice)
fc.get_tree()
# Predict
preds = fc.predict(X_test)
Score
fc.score(X_test, y_test)
```

# About the project

`fasttrees` is &copy; 2019-{{ "now" | date: "%Y" }} by [Dominic Zijlstra](https://www.linkedin.com/in/dominiczijlstra/) and [Stefan Bachhofner](https://www.linkedin.com/in/stefan-bachhofner-b729031b0/).

The implementation is inspired and based on on the `R` package [FFTrees](https://cran.r-project.org/web/packages/FFTrees/index.html), developed by Phillips, Neth, Woike and Grassmaier.

## License

`fasttrees` is diributed by an [MIT license](https://github.com/fasttrees/fasttrees/blob/master/LICENSE.txt).

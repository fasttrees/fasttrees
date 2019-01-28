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

## Usage
Instantiate a fast-and-frugal tree classifier:
```
fc = FastFrugalTreeClassifier()
```

Fit on your data:
```
fc.fit(X_train, y_train)
```

View the fitted tree (this is especially useful if the 'predictions' will be carried out by humans in practice):
```
fc.get_tree()
```

Predict:
```
preds = fc.predict(X_test)
```

Score:
```
fc.score(X_test, y_test)
```

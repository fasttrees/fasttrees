import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import balanced_accuracy_score
import operator


construction_algorithms = ['marginal_fan']
maximize_metrics = ['bacc']
operator_dict = {'<=': operator.le, '>': operator.gt, '==': operator.eq, 'in': lambda val, lst: val in lst}


class FastFrugalTreeClassifier(BaseEstimator, ClassifierMixin):
    """FFT classifier"""

    def __init__(self, construction_algorithm='marginal_fan', maximize_metric='bacc', max_levels=4, stopping_param=.1):
        """Inits FFT classifier.

        Args:
            construction_algorithm: algorithm used to create trees. Currently supported: 'marginal_fan'
            maximize_metric: metric to maximize when choosing threshold. Currently supported: 'bacc'
            max_levels: maximum number of levels for possible trees
            stopping_param: prune levels containing less than stopping_param of cases

        Returns:
            None
        """
        if construction_algorithm in construction_algorithms:
            self.construction_algorithm = construction_algorithm
        else:
            raise ValueError(
                'Not a valid construction_algorithm. Possible choices are {}'.format(construction_algorithms))

        if maximize_metric in maximize_metric:
            self.maximize_metric = maximize_metric
        else:
            raise ValueError('Not a valid maximize_metric. Possible choices are {}'.format(maximize_metric))

        self.max_levels = int(max_levels)

        self.stopping_param = float(stopping_param)

    def _score(self, y, predictions):
        if self.maximize_metric == 'bacc':
            return balanced_accuracy_score(y, predictions)

    def _get_thresholds(self, X, y):
        """Get thresholds and directions that maximimize maximize_metric for each feature.

        Args:
            X: Dataframe with features as columns
            y: true values

        Returns:
            Dataframe with row for every feature, with threshold, direction and maximize_metric, sorted by maximize_metric
        """
        threshold_df = pd.DataFrame(columns=['feature', 'type', 'threshold', 'direction', self.maximize_metric])
        for i, col in enumerate(X):
            threshold_df.loc[i, 'feature'] = col
            if X[col].dtype.name == 'category':
                # categorical
                threshold_df.loc[i, 'type'] = 'categorical'
                categories = X[col].cat.categories
                metric_max = 0
                for l in range(1, len(categories)):
                    for subset in itertools.combinations(categories, l):
                        predictions = X[col].isin(subset)
                        metric = self._score(y, predictions)
                        if metric >= metric_max:
                            metric_max = metric
                            direction_max = 'in'
                            threshold = subset
            else:
                # numerical
                threshold_df.loc[i, 'type'] = 'numerical'
                test_values = X[col].unique()
                metric_max = 0
                for val in test_values:
                    for direction, operator in {op: operator_dict[op] for op in ['<=', '>']}.items():
                        predictions = operator(X[col], val)
                        metric = self._score(y, predictions)
                        if metric >= metric_max:
                            metric_max = metric
                            direction_max = direction
                            threshold = val
            threshold_df.loc[i, 'threshold'] = threshold
            threshold_df.loc[i, 'direction'] = direction_max
            threshold_df.loc[i, self.maximize_metric] = metric_max
        threshold_df[self.maximize_metric] = threshold_df[self.maximize_metric].astype(float)
        self.thresholds = threshold_df.sort_values(by=self.maximize_metric, ascending=False).reset_index(drop=True)

    def _predict_all(self, X, cue_df):
        """Make predictions for X given cue_df.

        Args:
            X: Dataframe with features as columns
            cue_df: Dataframe with ordered features, directions, thresholds, exits

        Returns:
            Dataframe with row for every feature, with threshold, direction and maximize_metric, sorted by maximize_metric
        """
        nr_rows = cue_df.shape[0]

        def prediction_func(row):
            for index, cue_row in cue_df.iterrows():
                operator = operator_dict[cue_row['direction']]
                outcome = operator(row[cue_row['feature']], cue_row['threshold'])
                row.set_value(index, outcome)
                if ((cue_row['exit'] == 1) and outcome) or ((cue_row['exit'] == 0) and not outcome) or (
                        index + 1 == nr_rows):
                    cues_used = index + 1
                    break

            return row[-cues_used:]

        all_predictions = X.apply(prediction_func, axis=1)
        return all_predictions

    def _predict_and_prune(self, X, cue_df):
        """
            Make predictions and prune features that classify less than stopping_param
        """
        all_predictions = self._predict_all(X, cue_df)

        # prune non classifying features
        cols = [col for col in all_predictions if all_predictions[col].notnull().mean() >= self.stopping_param]

        all_predictions = all_predictions[cols]

        # get last prediction
        predictions = all_predictions.ffill(axis=1).iloc[:, -1]

        nr_cues_used = len(cols)

        return predictions, nr_cues_used

    def _growtrees(self, X, y):
        """
        Grow all possible trees up to self.max_levels. Prune levels classifying less than self.stopping_param
        """
        relevant_features = self.thresholds.head(self.max_levels)
        midx = pd.MultiIndex(levels=[[], []],
                             labels=[[], []],
                             names=['tree', 'idx'])
        tree_df = pd.DataFrame(columns=['feature', 'type', 'threshold', 'direction', self.maximize_metric], index=midx)
        for tree in range(2 ** (self.max_levels - 1)):
            for index, feature_row in relevant_features.iterrows():
                tree_df.loc[
                    (tree, index), ['feature', 'type', 'threshold', 'direction', self.maximize_metric]] = feature_row

                # exit 0.5 is stop, exit 1 means stop on true, exit 0 means stop on false
                tree_df.loc[(tree, index), 'exit'] = np.binary_repr(tree, width=self.max_levels)[-1 - index]

            tree_df['exit'] = tree_df['exit'].astype(float)

            predictions, nr_cues_used = self._predict_and_prune(X, tree_df.loc[tree])

            for i in range(nr_cues_used, self.max_levels):
                tree_df.drop(index=(tree, i), inplace=True)
            tree_df.loc[(tree, nr_cues_used - 1), 'exit'] = 0.5

            tree_df.loc[tree, self.maximize_metric] = self._score(y, predictions)

        self._all_trees = tree_df

    def fit(self, X, y):
        """
        Fit FFT
        """
        self._get_thresholds(X, y)
        self._growtrees(X, y)
        self._best_tree = self._all_trees[
            self._all_trees[self.maximize_metric] == self._all_trees[self.maximize_metric].max()]
        self._best_tree.index = self._best_tree.index.droplevel(level=0)
        return self

    def predict(self, X):
        all_predictions = self._predict_all(X, self._best_tree)
        return all_predictions.ffill(axis=1).iloc[:, -1]

    def score(self, X, y=None):
        return self._score(y, self.predict(X))
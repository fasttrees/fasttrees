import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import balanced_accuracy_score
import operator
import logging

construction_algorithms = ['marginal_fan']
operator_dict = {'<=': operator.le, '>': operator.gt, '==': operator.eq, 'in': lambda val, lst: val in lst}


class FastFrugalTreeClassifier(BaseEstimator, ClassifierMixin):
    """Fast-and-Frugal-Tree classifier

        Inits Fast-and-Frugal-Tree classifier.

        Parameters
        ----------
        construction_algorithm : str, default='marginal_fan'
            Specifies the algorithm used to create trees. Currently only supports 'marginal_fan'.

        scorer : func, default=sklearn.metrics.scorer.balanced_accuracy_score
            Specifies the metric to maximize when choosing threshold. Any function that returns higher values for better predictions.

        max_levels : int
            Specifies the maximum number of levels for possible trees.

        stopping_param : float
            Specifies the prune levels containing less than ``stopping_param`` of cases.

        max_categories : int
            Specifies the maximum number of categories to group together for categorical columns.

        max_cuts : int
            Specifies the maximum number of cuts to try on a numerical column.

        Examples
        ----------
        >>> from fasttrees.fasttrees import FastFrugalTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_features=4, random_state=0)
        >>> fc = FastFrugalTreeClassifier
        >>> fc.fit(X, y)
        >>> fc.get_tree()
    """

    def __init__(self, construction_algorithm='marginal_fan', scorer=balanced_accuracy_score, max_levels=4,
                 stopping_param=.1, max_categories=4, max_cuts=100):
        if construction_algorithm in construction_algorithms:
            self.construction_algorithm = construction_algorithm
        else:
            raise ValueError(
                'Not a valid construction_algorithm. Possible choices are {}'.format(construction_algorithms))

        self.scorer = scorer

        self.max_levels = int(max_levels)

        self.stopping_param = float(stopping_param)

        self.max_categories = max_categories

        self.max_cuts = max_cuts

    def _score(self, y, predictions):
        """
        Return the score on the given ``y`` and ``predictions``.

        Parameters
        ----------
            y : pandas.DataFrame
                The real outcomes.

            predictions : pandas.DataFrame
                The predicted outcomes.

        Returns
        ----------
            score : float
                The score w.r.t. ``y``.
        """
        return self.scorer(y, predictions)

    def _get_thresholds(self, X, y):
        """
        Get possible thresholds and directions for each feature.

        Parameters
        ----------
            X : pandas.DataFrame
                The test samples as a Dataframe with features as columns. Features can be numerical or categorical.
            y : pandas.DataFrame
                The true labels for ``X```, which are real, or binary, outcomes.

        Returns
        ----------
            self.all_thresholds : pandas.DataFrame
                A dataframe with rows for every feature, with threshold, direction and scorer.
        """
        midx = pd.MultiIndex(levels=[[], []],
                             labels=[[], []],
                             names=['cue_nr', 'threshold_nr'])
        threshold_df = pd.DataFrame(columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__],
                                    index=midx)

        # Get optimal classification threshold for each feature
        for i, col in enumerate(X):
            logging.debug('Get threshold for {}'.format(col))
            j = 0

            if X[col].dtype.name == 'category':
                # categorical
                categories = X[col].cat.categories

                threshold_df['threshold'] = threshold_df['threshold'].astype(object)

                # try all possible subsets of categories

                for l in range(1, min(len(categories), self.max_categories + 1)):
                    for subset in itertools.combinations(categories, l):
                        predictions = X[col].isin(subset)
                        metric = self._score(y, predictions)

                        # save metric, direction and threshold
                        threshold_df.at[(i, j), 'direction'] = 'in'
                        threshold_df.at[(i, j), 'threshold'] = subset
                        threshold_df.at[(i, j), self.scorer.__name__] = metric
                        j += 1

                threshold_df.loc[i, 'type'] = 'categorical'
            else:
                # numerical
                percentiles = np.linspace(0, 100, self.max_cuts + 1)

                test_values = np.percentile(X[col], percentiles)

                # try smaller than and bigger than for every unique value in column
                for val in test_values:
                    for direction, operator in {op: operator_dict[op] for op in ['<=', '>']}.items():
                        predictions = operator(X[col], val)
                        metric = self._score(y, predictions)

                        threshold_df.at[(i, j), 'threshold'] = val
                        threshold_df.at[(i, j), 'direction'] = direction
                        threshold_df.at[(i, j), self.scorer.__name__] = metric
                        j += 1

                threshold_df.loc[i, 'type'] = 'numerical'

            threshold_df.loc[i, 'feature'] = col

        threshold_df[self.scorer.__name__] = threshold_df[self.scorer.__name__].astype(float)

        # sort features by their score
        self.all_thresholds = threshold_df

    def _get_best_thresholds(self):
        """
        Get thresholds and directions that maximimize scorer for each feature.

        Returns
        ----------
            self.thresholds : pandas.DataFrame
                A dataframe with rows for every feature, with threshold, direction and scorer, sorted by scorer.
        """
        threshold_df = pd.DataFrame(columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__])
        for cue_nr, cue_df in self.all_thresholds.groupby(level=0):
            idx = cue_df[self.scorer.__name__].idxmax()
            threshold_df.loc[cue_nr, ['feature', 'direction', 'threshold', 'type', self.scorer.__name__]] = cue_df.loc[
                idx]

        threshold_df[self.scorer.__name__] = threshold_df[self.scorer.__name__].astype(float)

        self.thresholds = threshold_df.sort_values(by=self.scorer.__name__, ascending=False).reset_index(drop=True)

    @staticmethod
    def _predict_all(X, cue_df):
        """
        Make predictions for ``X`` given ``cue_df``.

        Parameters
        ----------
            X : pandas.Dataframe
                The input samples as a dataframe with features as columns. Features can be numerical or categorical.

            cue_df : pandas.Dataframe
                A dataframe with ordered features, directions, thresholds, and exists.

        Returns
        ----------
            all_predictions : pandas.Series
                A series with a prediction for every cue in cue_df up to the point where the fast-and-frugal-tree was exited.
        """
        nr_rows = cue_df.shape[0]

        # could be replaced with logical expression which would not have to be applied row-wise? currently very slow
        def prediction_func(row):
            """
            Makes a prediction for the given feature row.

            Look up the row's features in order of their score. Exit if the threshold is met and the tree exits on True,
            or if the threshold is not met and the tree exits on False.

            Parameters
            ----------
                row : dict
                    A dict with the features.

            Returns
            ----------
                ret_ser : pandas.Series
                    A series with a prediction for all cues used.
            """
            ret_ser = pd.Series()
            for index, cue_row in cue_df.iterrows():
                operator = operator_dict[cue_row['direction']]
                outcome = operator(row[cue_row['feature']], cue_row['threshold'])

                # store prediction in series
                ret_ser.set_value(index, outcome)

                # exit tree if outcome is exit or last cue reached
                if (cue_row['exit'] == int(outcome)) or (index + 1 == nr_rows):
                    cues_used = index + 1
                    break

            # return predictions for cues used
            return ret_ser

        all_predictions = X.apply(prediction_func, axis=1)
        return all_predictions

    @staticmethod
    def _get_final_prediction(all_predictions):
        """
        Get final (latest non-null) predictions from all cue predictions.

        Parameters
        ----------
            all_predictions : pandas.Dataframe
               A dataframe with all predictions.

        Returns
        ----------
            final_prediction : pandas.DataFrame
                A data frame with the final predictions.
        """
        return all_predictions.ffill(axis=1).iloc[:, -1]

    def _predict_and_prune(self, X, cue_df):
        """
        Make predictions and prune features that classify less than ``self.stopping_param``.

        Parameters
        ----------
            X : pandas.Dataframe
                The training input samples with features as columns. Features can be numerical or categorical.

        Returns
        ----------
            Tuple
                A tuple of length three where the first element are the predictions, the second element are the nr cused used, and the third ond
                are the fraction used.
        """
        logging.debug('Predicting ...')
        all_predictions = self._predict_all(X, cue_df)

        # prune non classifying features
        logging.debug('Pruning ...')
        fraction_used = all_predictions.notnull().mean()

        cols = [col for col in all_predictions if fraction_used[col] >= self.stopping_param]

        all_predictions = all_predictions[cols]
        fraction_used = fraction_used[:len(cols)]

        # get last prediction
        predictions = self._get_final_prediction(all_predictions)

        nr_cues_used = len(cols)

        return predictions, nr_cues_used, fraction_used

    def _growtrees(self, X, y):
        """
        Grow all possible trees up to ``self.max_levels``. Levels that classify less than``self.stopping_param`` are pruned.

        Parameters
        ----------
            X : pandas.Dataframe
                The training input samples with features as columns. Features can be numerical or categorical.

            y : pandas.Dataframe
                The target class labels as real or binary outcomes.

        Returns
        ----------
            self.all_trees : pandas.DataFrame
                A dataframe with all trees grown.
        """
        relevant_features = self.thresholds.head(self.max_levels)
        midx = pd.MultiIndex(levels=[[], []],
                             labels=[[], []],
                             names=['tree', 'idx'])
        tree_df = pd.DataFrame(
            columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__, 'fraction_used'], index=midx)
        for tree in range(2 ** (self.max_levels - 1)):
            logging.debug('Grow tree {}...'.format(tree))
            for index, feature_row in relevant_features.iterrows():
                tree_df['threshold'] = tree_df['threshold'].astype(object)

                tree_df.loc[
                    (tree, index), ['feature', 'direction', 'threshold', 'type', self.scorer.__name__]] = feature_row

                # exit 0.5 is stop, exit 1 means stop on true, exit 0 means stop on false
                tree_df.loc[(tree, index), 'exit'] = np.binary_repr(tree, width=self.max_levels)[-1 - index]

            tree_df['exit'] = tree_df['exit'].astype(float)

            predictions, nr_cues_used, fraction_used = self._predict_and_prune(X, tree_df.loc[tree])

            for i in range(nr_cues_used, self.max_levels):
                tree_df.drop(index=(tree, i), inplace=True)

            tree_df.loc[tree, 'fraction_used'] = fraction_used.values
            tree_df.loc[(tree, nr_cues_used - 1), 'exit'] = 0.5

            score = self._score(y, predictions)
            logging.debug('Score is {}...'.format(score))
            tree_df.loc[tree, self.scorer.__name__] = score

        self.all_trees = tree_df

    def get_tree(self, idx=None, decision_view=True):
        """
        Get tree with index ``idx`` from all trees.

        Parameters
        ----------
            idx : int, Default=None
                The index of the desired tree. Default is None, which returns the best tree.

            decision_view : bool, default=True
                If true, it will return a dataframe in an easily readable form, which can then be used to make a quick decision.
                If false, it will return the original dataframe with more statistics.
                The default is ``True``.

        Returns
        ----------
            tree_df : pandas.DataFrame
                The dataframe of the tree with index ``idx``.
        """
        if idx is None:
            idx = self.all_trees[self.scorer.__name__].idxmax()[0]

        tree_df = self.all_trees.loc[idx]

        if decision_view:
            def exit_action(exit):
                ret_ser = pd.Series()
                ret_ser.set_value('IF YES', '↓')
                ret_ser.set_value('IF NO', '↓')
                if exit <= 0.5:
                    ret_ser.set_value('IF NO', 'decide NO')
                if exit >= 0.5:
                    ret_ser.set_value('IF YES', 'decide YES')
                return ret_ser

            tree_df = pd.concat([tree_df, tree_df['exit'].apply(exit_action)], axis=1)
            tree_df = tree_df[['IF NO', 'feature', 'direction', 'threshold', 'IF YES']]

        return tree_df

    def fit(self, X, y):
        """
        Builds the fast and frugal tree classifier from the training set (X, y).

        Parameters
        ----------
            X : pandas.Dataframe
                The training input samples with features as columns. Features can be numerical or categorical.

            y : pandas.Dataframe
                The target class labels as real or binary outcomes.

        Returns
        ----------
            self : FastFrugalTreeClassifier
                Fitted estimator.
        """
        self._get_thresholds(X, y)
        self._get_best_thresholds()
        self._growtrees(X, y)
        self.best_tree = self.get_tree()
        return self

    def predict(self, X, tree_idx=None):
        """
        Predict class value for ``X``.

        Returns the predicted class for each sample in ``X``.

        Parameters
        ----------
            X : pandas.DataFrame
                The input samples as a Dataframe with features as columns. Features can be numerical or categorical.

            tree_idx : int, default=None
                The tree to use for the predictions. Default is best tree.

        Returns
        ----------
           y : pandas.DataFrame
                The predicted classes.
        """
        all_predictions = self._predict_all(X, self.get_tree(tree_idx, decision_view=False))
        return self._get_final_prediction(all_predictions)

    def score(self, X, y=None):
        """
        Predicts for data X. Scores predictions against y.

        Parameters
        ----------
            X : pandas.DataFrame
                The test samples as a Dataframe with features as columns. Features can be numerical or categorical.

            y : pandas.DataFrame, default=None
                The true labels for ``X```.

        Returns
        ----------
            score : float
                The score of ``self.predict(X)`` w.r.t ``y``.
        """
        return self._score(y, self.predict(X))

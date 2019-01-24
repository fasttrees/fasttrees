import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import balanced_accuracy_score
import operator

construction_algorithms = ['marginal_fan']
operator_dict = {'<=': operator.le, '>': operator.gt, '==': operator.eq, 'in': lambda val, lst: val in lst}


class FastFrugalTreeClassifier(BaseEstimator, ClassifierMixin):
    """Fast-and-Frugal-Tree classifier"""

    def __init__(self, construction_algorithm='marginal_fan', scorer=balanced_accuracy_score, max_levels=4,
                 stopping_param=.1):
        """Inits Fast-and-Frugal-Tree classifier.

        Args:
            construction_algorithm: algorithm used to create trees. Currently supported: 'marginal_fan'
            scorer: metric to maximize when choosing threshold. Any function that returns higher values for better predictions
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

        self.scorer = scorer

        self.max_levels = int(max_levels)

        self.stopping_param = float(stopping_param)

    def _score(self, y, predictions):
        """Scores predictions against y.

        Args:
            y: real outcomes
            predictions: predicted outcomes

        Returns:
            Score
        """
        return self.scorer(y, predictions)

    def _get_thresholds(self, X, y):
        """Get possible thresholds and directions for each feature.

        Args:
            X: Dataframe with features as columns. Features can be numerical or categorical
            y: real, binary, outcomes.

        Returns:
            self.all_thresholds: Dataframe with rows for every feature, with threshold, direction
            and scorer
        """
        midx = pd.MultiIndex(levels=[[], []],
                             labels=[[], []],
                             names=['cue_nr', 'threshold_nr'])
        threshold_df = pd.DataFrame(columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__],
                                    index=midx)

        # Get optimal classification threshold for each feature
        for i, col in enumerate(X):
            metric_max = 0
            j = 0

            if X[col].dtype.name == 'category':
                # categorical
                categories = X[col].cat.categories

                threshold_df['threshold'] = threshold_df['threshold'].astype(object)

                # try all possible subsets of categories
                for l in range(1, len(categories)):
                    for subset in itertools.combinations(categories, l):
                        predictions = X[col].isin(subset)
                        metric = self._score(y, predictions)

                        # save metric, direction and threshold if they are the best so far
                        threshold_df.loc[(i, j), 'direction'] = 'in'
                        threshold_df.loc[(i, j), 'threshold'] = subset
                        threshold_df.loc[(i, j), self.scorer.__name__] = metric

                threshold_df.loc[i, 'type'] = 'categorical'
            else:
                # numerical
                test_values = X[col].unique()

                # try smaller than and bigger than for every unique value in column
                for val in test_values:
                    for direction, operator in {op: operator_dict[op] for op in ['<=', '>']}.items():
                        predictions = operator(X[col], val)
                        metric = self._score(y, predictions)

                        threshold_df.loc[(i, j), 'threshold'] = val
                        threshold_df.loc[(i, j), 'direction'] = direction
                        threshold_df.loc[(i, j), self.scorer.__name__] = metric
                        j += 1

                threshold_df.loc[i, 'type'] = 'numerical'

            threshold_df.loc[i, 'feature'] = col

        threshold_df[self.scorer.__name__] = threshold_df[self.scorer.__name__].astype(float)

        # sort features by their score
        self.all_thresholds = threshold_df  # .sort_values(by=['cue_nr' , self.scorer.__name__], ascending=False)#.reset_index(drop=True)

    def _get_best_thresholds(self):
        """Get thresholds and directions that maximimize scorer for each feature.

        Args:

        Returns:
            self.thresholds: Dataframe with rows for every feature, with threshold, direction
            and scorer, sorted by scorer
        """
        threshold_df = pd.DataFrame(columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__])
        for cue_nr, cue_df in self.all_thresholds.groupby(level=0):
            idx = cue_df['balanced_accuracy_score'].idxmax()
            threshold_df.loc[cue_nr, ['feature', 'direction', 'threshold', 'type', self.scorer.__name__]] = cue_df.loc[
                idx]

        threshold_df[self.scorer.__name__] = threshold_df[self.scorer.__name__].astype(float)

        self.thresholds = threshold_df.sort_values(by=self.scorer.__name__, ascending=False).reset_index(drop=True)

    @staticmethod
    def _predict_all(X, cue_df):
        """Make predictions for X given cue_df.

        Args:
            X: Dataframe with features as columns
            cue_df: Dataframe with ordered features, directions, thresholds, exits

        Returns:
            Series with prediction for every cue in cue_df up to the point the fast-and-frugal-tree was exited
        """
        nr_rows = cue_df.shape[0]

        def prediction_func(row):
            """Look up the row's features in order of their score. Exit if the threshold is met and the tree exits on True,
            or if the threshold is not met and the tree exits on False.

            Args:
                row: Dataframe row with features as columns

            Returns:
                Series with prediction for all cues used
            """
            for index, cue_row in cue_df.iterrows():
                operator = operator_dict[cue_row['direction']]
                outcome = operator(row[cue_row['feature']], cue_row['threshold'])

                # store prediction in series
                row.set_value(index, outcome)

                # exit tree if outcome is exit or last cue reached
                if (cue_row['exit'] == int(outcome)) or (index + 1 == nr_rows):
                    cues_used = index + 1
                    break

            # return predictions for cues used
            return row[-cues_used:]

        all_predictions = X.apply(prediction_func, axis=1)
        return all_predictions

    @staticmethod
    def _get_final_prediction(all_predictions):
        """Get final (latest non-null) predictions from all cue predictions.

        Args:
            X: Dataframe with all predictions

        Returns:
            Dataframe with final prediction
        """
        return all_predictions.ffill(axis=1).iloc[:, -1]

    def _predict_and_prune(self, X, cue_df):
        """Make predictions and prune features that classify less than stopping_param.

        Args:
            X: Dataframe with all predictions

        Returns:
            Dataframe with pruned prediction
        """
        all_predictions = self._predict_all(X, cue_df)

        # prune non classifying features
        cols = [col for col in all_predictions if all_predictions[col].notnull().mean() >= self.stopping_param]

        all_predictions = all_predictions[cols]

        # get last prediction
        predictions = self._get_final_prediction(all_predictions)

        nr_cues_used = len(cols)

        return predictions, nr_cues_used

    def _growtrees(self, X, y):
        """Grow all possible trees up to self.max_levels. Prune levels classifying less than self.stopping_param

        Args:
            X: Dataframe with features as columns. Features can be numerical or categorical
            y: real, binary, outcomes.

        Returns:
            self.all_trees: Dataframe with all trees grown
        """
        relevant_features = self.thresholds.head(self.max_levels)
        midx = pd.MultiIndex(levels=[[], []],
                             labels=[[], []],
                             names=['tree', 'idx'])
        tree_df = pd.DataFrame(columns=['feature', 'direction', 'threshold', 'type', self.scorer.__name__], index=midx)
        for tree in range(2 ** (self.max_levels - 1)):
            for index, feature_row in relevant_features.iterrows():
                tree_df['threshold'] = tree_df['threshold'].astype(object)

                tree_df.loc[
                    (tree, index), ['feature', 'direction', 'threshold', 'type', self.scorer.__name__]] = feature_row

                # exit 0.5 is stop, exit 1 means stop on true, exit 0 means stop on false
                global testvar
                testvar = tree_df
                tree_df.loc[(tree, index), 'exit'] = np.binary_repr(tree, width=self.max_levels)[-1 - index]

            tree_df['exit'] = tree_df['exit'].astype(float)

            predictions, nr_cues_used = self._predict_and_prune(X, tree_df.loc[tree])

            for i in range(nr_cues_used, self.max_levels):
                tree_df.drop(index=(tree, i), inplace=True)
            tree_df.loc[(tree, nr_cues_used - 1), 'exit'] = 0.5

            tree_df.loc[tree, self.scorer.__name__] = self._score(y, predictions)

        self.all_trees = tree_df

    def get_tree(self, idx=None):
        """Get specific tree from all trees

        Args:
            idx: index of desired tree. Will return best tree if None

        Returns:
            Dataframe of tree
        """
        if idx is None:
            idx = self.all_trees[self.scorer.__name__].idxmax()[0]
        return self.all_trees.loc[idx]

    def fit(self, X, y):
        """Fits the classifier to the data.

        Args:
            X: Dataframe with features as columns. Features can be numerical or categorical
            y: real, binary, outcomes.

        Returns:
            self: Fitted FastFrugalTreeClassifier
        """
        self._get_thresholds(X, y)
        self._get_best_thresholds()
        self._growtrees(X, y)
        self.best_tree = self.get_tree()
        return self

    def predict(self, X, tree_idx=None):
        """Predicts outcomes for data X.

        Args:
            X: Dataframe with features as columns. Features can be numerical or categorical
            tree_idx: tree to use, default is best tree

        Returns:
            predictions
        """
        all_predictions = self._predict_all(X, self.get_tree(tree_idx))
        return self._get_final_prediction(all_predictions)

    def score(self, X, y=None):
        """Predicts for data X. Scores predictions against y.

        Args:
            X: Dataframe with features as columns. Features can be numerical or categorical
            y: real outcomes

        Returns:
            Score
        """
        return self._score(y, self.predict(X))
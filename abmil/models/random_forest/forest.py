from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from models.random_forest.base import EstimatorParamsMixin
from models.random_forest.tree import DecisionTree


@dataclass
class RandomForest(EstimatorParamsMixin, BaseEstimator, ClassifierMixin):
    estimators_num: int = 100

    def fit(self, X_train: np.array, y_train: np.array) -> RandomForest:
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(
            X_train, y_train, ensure_2d=True, allow_nd=True, estimator=self
        )

        # Store the trees and fit them
        self.classes_ = self.classes or unique_labels(y_train) # noqa: Required by sklearn
        self.trees = [  # noqa: Required by sklearn
            DecisionTree(
                max_features=self.max_features,
                classes=self.classes,
                dim=self.dim,
                max_node_depth=self.max_node_depth,
                min_node_element_count=self.min_node_element_count,
            ).fit(X_train, y_train)
            for _ in range(self.estimators_num)
        ]
        # Return the classifier
        return self

    @property
    def oob_score(self) -> float:
        check_is_fitted(self)
        return np.float(np.mean([tree.oob_score for tree in self.trees]))

    def predict(self, X_test: np.array) -> np.array:
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        return np.asarray(
            [
                np.argmax(np.bincount([tree.predict([x])[0] for tree in self.trees]))
                for x in X_test
            ]
        )

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from models.random_forest.base import (
    BaseDecisionTree,
    BaseMIDecisionTree,
)
from models.random_forest.node import Node, MINode, NodeMixin
from models.random_forest.validators import (
    NodeMaxDepthValidator,
    NodeMinElementCountValidator,
)


@dataclass
class DecisionTree(
    BaseDecisionTree, BaseEstimator, ClassifierMixin
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_validators = [
            NodeMaxDepthValidator(self.max_node_depth),
            NodeMinElementCountValidator(self.min_node_element_count),
        ]

    def draw_bootstrap(self, X_train: np.array, y_train: np.array) -> None:
        bootstrap_indices = indices = np.arange(len(X_train))
        while len(np.unique(bootstrap_indices)) == len(indices):
            bootstrap_indices = np.random.choice(indices, len(indices), replace=True)

        oob_mask = ~np.isin(indices, bootstrap_indices)

        self.X_bootstrap = X_train[bootstrap_indices]
        self.y_bootstrap = y_train[bootstrap_indices]

        self.X_oob = X_train[oob_mask]
        self.y_oob = y_train[oob_mask]

    def fit(self, X_train: np.array, y_train: np.array) -> DecisionTree:

        self._process_train_data(X_train, y_train)

        self.draw_bootstrap(X_train, y_train)
        self.root_ = Node(self, np.asarray([True] * len(self.X_bootstrap)),)  # noqa: Required by sklearn
        self.build()
        # Return the classifier
        return self

    def predict(self, X_test: np.array) -> np.array:
        X_test = self._process_test_data(X_test)

        return np.asarray([self.__predict_node(x, self.root_) for x in X_test])

    def __predict_node(self, x: np.array, node: Node) -> int:
        if node.is_terminal:
            return node.prediction
        feature_id, feature_value = node.split
        if x[feature_id] <= feature_value:
            return self.__predict_node(x, node.left)
        return self.__predict_node(x, node.right)

    @property
    def oob_score(self) -> float:
        check_is_fitted(self)
        if self.X_oob.size == 0:
            raise ValueError("The X_oob must not be empty.")
        return np.float(np.sum(~(self.predict(self.X_oob) == self.y_oob))) / len(
            self.X_oob
        )

    def _validate_node(self, node: NodeMixin) -> bool:
        return all(validator.validate(node) for validator in self.node_validators)

    def _split_node(self, node: Node) -> None:
        if not self._validate_node(node):
            return None
        selected_features_indices = np.random.choice(
            np.arange(self.dim), replace=False, size=self.max_features
        )
        best_gain = -node.entropy
        for feature_id in selected_features_indices:
            for split_value in np.unique(self.X_bootstrap[node.mask, feature_id]):
                split = (feature_id, split_value)
                left = Node(
                    self,
                    node.mask & (self.X_bootstrap[:, feature_id] <= split_value),
                    depth=node.depth + 1,
                )
                right = Node(
                    self,
                    node.mask & (self.X_bootstrap[:, feature_id] > split_value),
                    depth=node.depth + 1,
                )
                if not (self._validate_node(left) and self._validate_node(right)):
                    continue
                split_gain = node.gain(left, right)
                if split_gain > best_gain:
                    best_gain = split_gain
                    node.left = left
                    node.right = right
                    node.split = split

        if node.is_terminal:
            return None
        self._split_node(node.left)
        self._split_node(node.right)

    def build(self):
        self._split_node(self.root_)


@dataclass
class MIDecisionTree(BaseMIDecisionTree, DecisionTree):
    def __init__(self, *args, **kwargs):
        super(BaseMIDecisionTree, self).__init__(*args, **kwargs)

    @staticmethod
    def __make_bag_labels(bag_indices_ranges: np.array, y: np.array) -> np.array:
        return np.asarray([np.max(y[i[0] : i[1]]) for i in bag_indices_ranges])

    def draw_bootstrap(
        self,
        X_train: np.array,
        y_train: np.array,
        X_bags_indices_ranges_train: np.array,
    ) -> None:
        bootstrap_indices = indices = np.arange(len(X_bags_indices_ranges_train))
        while len(np.unique(bootstrap_indices)) == len(indices):
            bootstrap_indices = np.random.choice(indices, len(indices), replace=True)

        oob_mask = ~np.isin(indices, bootstrap_indices)

        self.X_bags_indices_ranges_bootstrap = X_bags_indices_ranges_train[
            bootstrap_indices
        ]
        self.y_bags_indices_ranges_bootstrap = self.__make_bag_labels(
            self.X_bags_indices_ranges_bootstrap, y_train
        )

        self.X_bags_indices_ranges_oob = X_bags_indices_ranges_train[oob_mask]
        self.y_bags_indices_ranges_oob = self.__make_bag_labels(
            self.X_bags_indices_ranges_oob, y_train
        )

        self.X_instances_train = X_train
        self.y_instances_train = y_train

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_bags_indices_ranges_train: np.array,
    ) -> MIDecisionTree:

        self._process_train_data(X_train, y_train)

        self.draw_bootstrap(X_train, y_train, X_bags_indices_ranges_train)
        self.root_ = MINode(  # noqa: Required by sklearn
            self, np.asarray([True] * len(self.X_bags_indices_ranges_bootstrap)),
        )
        self.build()
        # Return the classifier
        return self

    def predict(
        self, X_test: np.array, X_bags_indices_ranges_test: np.array
    ) -> np.array:

        X_test = self._process_test_data(X_test)

        return np.asarray(
            [
                self.__predict_node(X_test, bag_indices_range, self.root_)
                for bag_indices_range in X_bags_indices_ranges_test
            ]
        )

    def __predict_node(
        self, X_test: np.array, bag_indices_range: np.array, node: MINode
    ) -> int:
        if node.is_terminal:
            return node.prediction
        feature_id, feature_value = node.split
        instances = X_test[bag_indices_range[0] : bag_indices_range[1]]
        if np.sum(instances[:, feature_id] <= feature_value) > 0:
            return self.__predict_node(X_test, bag_indices_range, node.left)
        return self.__predict_node(X_test, bag_indices_range, node.right)

    @property
    def oob_score(self) -> float:
        check_is_fitted(self)
        if self.X_bags_indices_ranges_oob.size == 0:
            raise ValueError("The X_bags_indices_ranges_oob must not be empty.")
        return np.float(
            np.sum(
                ~(
                    self.predict(self.X_instances_train, self.X_bags_indices_ranges_oob)
                    == self.y_bags_indices_ranges_oob
                )
            )
        ) / len(self.X_bags_indices_ranges_oob)

    def _split_node(self, node: MINode) -> None:
        if not self._validate_node(node):
            return None
        selected_features_indices = np.random.choice(
            np.arange(self.dim), replace=False, size=self.max_features
        )
        best_gain = -node.entropy
        X_node_indices_ranges, _ = node.elements
        X = np.vstack(
            [self.X_instances_train[i[0]: i[1]] for i in X_node_indices_ranges]
        )
        node_bag_indices = np.where(node.mask)[0]
        for feature_id in selected_features_indices:
            for split_value in np.unique(X[:, feature_id]):
                split = (feature_id, split_value)
                bags_mask = np.ones(
                    len(self.X_bags_indices_ranges_bootstrap), dtype="int8"
                )
                for idx in node_bag_indices:
                    bag = self.X_bags_indices_ranges_bootstrap[idx]
                    bags_mask[idx] = np.int(
                        np.sum(
                            self.X_instances_train[bag[0] : bag[1], feature_id]
                            <= split_value
                        )
                        > 0
                    )

                left = MINode(self, node.mask & bags_mask, depth=node.depth + 1)
                right = MINode(self, node.mask & (~bags_mask), depth=node.depth + 1)
                if not (self._validate_node(left) and self._validate_node(right)):
                    continue
                split_gain = node.gain(left, right)
                if split_gain > best_gain:
                    best_gain = split_gain
                    node.left = left
                    node.right = right
                    node.split = split

        if node.is_terminal:
            return None
        self._split_node(node.left)
        self._split_node(node.right)

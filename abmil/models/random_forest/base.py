from dataclasses import dataclass, field
from typing import Optional, NamedTuple
from abc import ABC, abstractmethod
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array


@dataclass
class EstimatorParamsMixin:
    max_features: Optional[int] = None
    classes: Optional[int] = None
    dim: Optional[int] = None
    max_node_depth: int = 10
    min_node_element_count: int = 2

    def _process_train_data(self, X_train: np.array, y_train: np.array):
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(
            X_train, y_train, ensure_2d=True, allow_nd=True, estimator=self
        )
        # Store the classes and input dimension seen during fit
        self.classes = self.classes or unique_labels(y_train)
        self.dim = self.dim or X_train.shape[1]
        self.max_features = self.max_features or np.int(np.round(np.sqrt(self.dim)))
        self.max_features = min(self.max_features, self.dim)

    def _process_test_data(self, X_test: np.array) -> np.array:
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        return check_array(X_test)


class BaseDecisionTree(ABC, EstimatorParamsMixin):
    X_bootstrap: np.array = field(default_factory=lambda: np.asarray([]))
    y_bootstrap: np.array = field(default_factory=lambda: np.asarray([]))
    X_oob: np.array = field(default_factory=lambda: np.asarray([]))
    y_oob: np.array = field(default_factory=lambda: np.asarray([]))

    @abstractmethod
    def draw_bootstrap(self, X_train: np.array, y_train: np.array) -> None:
        ...


class BaseMIDecisionTree(ABC, EstimatorParamsMixin):
    X_bags_indices_ranges_bootstrap: np.array = field(
        default_factory=lambda: np.asarray([])
    )
    y_bags_indices_ranges_bootstrap: np.array = field(
        default_factory=lambda: np.asarray([])
    )
    X_bags_indices_ranges_oob: np.array = field(default_factory=lambda: np.asarray([]))
    y_bags_indices_ranges_oob: np.array = field(default_factory=lambda: np.asarray([]))
    X_instances_train: np.array = field(default_factory=lambda: np.asarray([]))
    y_instances_train: np.array = field(default_factory=lambda: np.asarray([]))

    @abstractmethod
    def draw_bootstrap(
            self, X_train: np.array, y_train: np.array, bags_indices_train: np.array
    ) -> None:
        ...


class ContinuousFeatureSplit(NamedTuple):
    feature_id: int
    feature_value: float

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from models.random_forest.base import (
    BaseDecisionTree,
    ContinuousFeatureSplit,
    BaseMIDecisionTree,
)


class NodeMixin:
    tree: BaseDecisionTree
    mask: np.array = field(default_factory=lambda: np.asarray([]))
    left: Optional[Node] = None
    right: Optional[Node] = None
    split: Optional[ContinuousFeatureSplit] = None
    depth: int = 0

    def __len__(self):
        return np.sum(self.mask)

    def __bool__(self):
        return len(self) > 0

    @property
    def elements(self) -> Tuple[np.array, np.array]:
        return self.tree.X_bootstrap[self.mask], self.tree.y_bootstrap[self.mask]

    @property
    def p(self) -> np.array:
        if not self:
            return np.zeros(self.tree.classes)
        _, labels = self.elements
        counts = np.bincount(labels)
        return counts / np.sum(counts)

    @property
    def entropy(self) -> float:
        p = self.p
        if np.sum(p == 0):
            return 0
        return -np.sum(p * np.log(p))

    def gain(self, left: NodeMixin, right: NodeMixin) -> Optional[float]:
        if not self:
            return 0
        return (
            self.entropy
            - len(left) / len(self) * left.entropy
            - len(right) / len(self) * right.entropy
        )

    @property
    def is_terminal(self) -> bool:
        return (self.left is None) and (self.right is None)

    @property
    def prediction(self) -> int:
        if not self.is_terminal:
            raise ValueError("An attempt to get a prediction from a non-terminal node.")
        return np.int(np.argmax(self.p))


@dataclass
class Node(NodeMixin):
    tree: BaseDecisionTree
    mask: np.array = field(default_factory=lambda: np.asarray([]))
    left: Optional[Node] = None
    right: Optional[Node] = None
    split: Optional[ContinuousFeatureSplit] = None
    depth: int = 0


@dataclass
class MINode(NodeMixin):
    tree: BaseMIDecisionTree
    mask: np.array = field(default_factory=lambda: np.asarray([]))
    left: Optional[MINode] = None
    right: Optional[MINode] = None
    split: Optional[ContinuousFeatureSplit] = None
    depth: int = 0

    @property
    def elements(self) -> Tuple[np.array, np.array]:
        return (
            self.tree.X_bags_indices_ranges_bootstrap[self.mask],
            self.tree.y_bags_indices_ranges_bootstrap[self.mask],
        )

from abc import ABC, abstractmethod
from dataclasses import dataclass

from models.random_forest.node import NodeMixin


class BaseNodeValidator(ABC):
    @abstractmethod
    def validate(self, node: NodeMixin) -> bool:
        ...


@dataclass
class NodeMaxDepthValidator(BaseNodeValidator):
    max_depth: int

    def validate(self, node: NodeMixin) -> bool:
        return node.depth <= self.max_depth


@dataclass
class NodeMinElementCountValidator(BaseNodeValidator):
    min_element_count: int

    def validate(self, node: NodeMixin) -> bool:
        return len(node) >= self.min_element_count

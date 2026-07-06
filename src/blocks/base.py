from abc import ABC, abstractmethod
from typing import Generic

from src.layers.base import Tensor


class Block(ABC, Generic[Tensor]):
    def __call__(
        self,
        inputs: Tensor,
        *,
        training: bool = False,
    ) -> Tensor:
        return self.forward(inputs, training=training)

    @abstractmethod
    def forward(
        self,
        inputs: Tensor,
        *,
        training: bool = False,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(
        self,
        grad: Tensor,
    ) -> Tensor:
        raise NotImplementedError   
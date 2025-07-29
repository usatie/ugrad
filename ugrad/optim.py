from ugrad import Tensor
from typing import Self, List


class SGD:
    def __init__(self, params: List[Tensor], lr: float, momentum: float) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = [Tensor.zeros_like(p) for p in params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for p, v in zip(self.params, self.v):
            v.data = self.momentum * v.data - self.lr * p.grad.data
            p.data += v.data

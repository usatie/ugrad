from ugrad import Tensor
from typing import Self, List


class SGD:
    def __init__(self, params: List[Tensor], lr: float, momentum: float, weight_decay: float = .0, nesterov: bool = False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.v = [Tensor.zeros_like(p) for p in params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for w, v in zip(self.params, self.v):
            g = w.grad + w * self.wd
            v.assign(self.momentum * v - self.lr * g)
            if self.nesterov:
                w.assign(w + self.momentum * v - self.lr * g)
            else:
                w.assign(w + v)

from ugrad import Tensor
from typing import Self, List

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float,
        momentum: float,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.v = [Tensor.zeros_like(p) for p in params]

    def step(self) -> None:
        for w, v in zip(self.params, self.v):
            g = w.grad + w * self.wd
            v.assign(self.momentum * v - self.lr * g)
            if self.nesterov:
                w.assign(w + self.momentum * v - self.lr * g)
            else:
                w.assign(w + v)

class Adam(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float=0.001,
        b1: float=0.9,
        b2: float=0.999,
        eps: float=1e-8
    ):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m0 = [Tensor.zeros_like(p) for p in self.params]
        self.v0 = [Tensor.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for w, m, v in zip(self.params, self.m0, self.v0):
            g = w.grad
            m.assign(self.b1 * m + (1-self.b1) * g)
            v.assign(self.b2 * v + (1-self.b2) * g**2)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            w.assign(w - self.lr * m_hat / (v_hat.sqrt() + self.eps))

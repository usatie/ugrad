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
        weight_decay: float = 0.01,
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


def Adam(
    params: List[Tensor],
    lr: float = 0.001,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
):
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


def AdamW(
    params: List[Tensor],
    lr: float = 0.001,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    return LAMB(params, lr, b1, b2, eps, weight_decay, adam=True)


def clamp(x: float, minval: float, maxval: float) -> float:
    return min(max(x, minval), maxval)


class LAMB(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adam: bool = False,
    ):
        super().__init__(params, lr)
        self.b1, self.b2, self.wd, self.eps = b1, b2, weight_decay, eps
        self.m0 = [Tensor.zeros_like(p) for p in self.params]
        self.v0 = [Tensor.zeros_like(p) for p in self.params]
        self.t = 0
        self.adam = adam

    def step(self) -> None:
        self.t += 1
        for w, m, v in zip(self.params, self.m0, self.v0):
            g = w.grad
            m.assign(self.b1 * m + (1 - self.b1) * g)
            v.assign(self.b2 * v + (1 - self.b2) * g**2)
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            r = m_hat / (v_hat.sqrt() + self.eps)
            update = r + self.wd * w
            if self.adam:
                trust_ratio = 1.0
            else:
                _ratio = w.square().sum().sqrt() / update.square().sum().sqrt()
                trust_ratio = clamp(_ratio.data.item(), 1.0, 10.0)
            w.assign(w - self.lr * trust_ratio * update)

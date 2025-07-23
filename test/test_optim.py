import numpy as np
import ugrad
import torch
from helper import ComparableTensor as Tensor
from helper import ComparableSGD as SGD


class MicroModel:
    def __init__(self):
        self.W1 = Tensor(np.random.randn(4, 8), requires_grad=True)
        self.W2 = Tensor(np.random.randn(8, 4), requires_grad=True)
        self.W3 = Tensor(np.random.randn(4, 1), requires_grad=True)

    def forward(self, x):
        return x.matmul(self.W1).relu().matmul(self.W2).relu().matmul(self.W3)

    def parameters(self):
        return [self.W1, self.W2, self.W3]


def test_SGD():
    x = Tensor(np.random.randn(32, 4))
    y = Tensor(np.random.randint(0, 2, 32).reshape(32, 1))
    model = MicroModel()

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    def loss_fn(output, target):
        squared_error = (target - output) ** 2
        batch_size = target.shape[0]
        return squared_error.sum() * (1 / batch_size)

    for i in range(1000):
        optimizer.zero_grad()
        out = model.forward(x)
        out.assert_all()
        loss = loss_fn(out, y)
        loss.assert_all()
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            p.assert_all()
    assert loss.data < 10.0

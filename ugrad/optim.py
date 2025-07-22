from ugrad import Tensor

class SGD:
    def __init__(self, params, lr, momentum):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = [Tensor.zeros_like(p) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p, v in zip(self.params, self.v):
            v.data = (self.momentum * v.data - self.lr * p.grad.data)
            p.data += v.data

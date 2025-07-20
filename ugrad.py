from typing import Self
import numpy as np

def register(F, name):
    def call(self, *args):
        f = F()  # Create fresh instance for each call
        args = (Tensor(x) if isinstance(x, int|float) else x for x in args)
        return Tensor(f(self, *args), func=f, requires_grad=f.requires_grad())
    setattr(Tensor, name, call)

class Tensor:
    def __init__(self, data, requires_grad=False, func=None):
        self.data = data
        self.func = func
        self.grad = None
        self.requires_grad=requires_grad
        self._backward = lambda x: ()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __radd__(self, other: Self|int|float):
        return self + other

    def __rmul__(self, other: Self|int|float):
        return self * other

    # self - other
    def __sub__(self, other: Self):
        return self + (-other)

    # other - self
    def __rsub__(self, other: Self):
        return (-self) + other

    def numpy(self):
        return self.data

    def backward(self):
        if self.grad is None and self.data.size == 1:
            self.grad = 1.0
        if self.func is None: return
        # Initialize inputs grads
        for t in self.func.inputs:
            if t.grad is None and t.requires_grad:
                t.grad = np.zeros_like(t.data)
        # update inputs grads
        grads = self.func.backward(self.grad)
        grads = grads if isinstance(grads, tuple) else (grads, )
        for t, g in zip(self.func.inputs, grads):
            if t.requires_grad:
                t.grad += g
        # recursively backward()
        for t in self.func.inputs:
            t.backward()

class Function:
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __call__(self, *args, **kwargs):
        self.inputs = args
        return self.forward(*args, **kwargs)

    def requires_grad(self):
        return any([t.requires_grad for t in self.inputs])

class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def backward(self, out_grad: Tensor):
        return out_grad, out_grad

register(Add, '__add__')

class Neg(Function):
    def forward(self, x: Tensor):
        return -x.data

    def backward(self, out_grad: Tensor):
        return -out_grad

register(Neg, '__neg__')

class Mul(Function):
    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def backward(self, out_grad: Tensor):
        x, y = self.inputs
        x_grad = out_grad * y.data
        y_grad = out_grad * x.data
        return x_grad, y_grad

register(Mul, '__mul__')

class Matmul(Function):
    def forward(self, x: Tensor, y: Tensor):
        # x (2, 3)
        # y (3, 4)
        # out (2, 4)
        out = x.data.dot(y.data)
        return out

    def backward(self, out_grad: Tensor):
        # x (2, 3)
        # y (3, 4)
        x, y = self.inputs
        # out_grad (2, 4)
        x_grad = out_grad.dot(y.data.transpose())
        y_grad = x.data.transpose().dot(out_grad)
        return x_grad, y_grad

register(Matmul, 'matmul')

class Sum(Function):
    def forward(self, x):
        return x.data.sum()

    def backward(self, out_grad):
        return out_grad

register(Sum, 'sum')

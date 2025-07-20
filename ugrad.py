from typing import Self
import numpy as np

class Tensor:
    def __init__(self, data, func=None):
        self.data = data
        self.func = func
        self.grad = None
        self._backward = lambda x: ()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other: Self|int|float):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        func = Add()
        out = Tensor(func(self, other), func=func)
        return out

    def __radd__(self, other: Self|int|float):
        return self + other

    def __mul__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        func = Mul()
        out = Tensor(func(self, other), func=func)
        return out

    def __rmul__(self, other: Self|int|float):
        return self * other

    # self - other
    def __sub__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        return Tensor(self.data - other.data)

    # other - self
    def __rsub__(self, other: Self):
        return (-self) + other

    # -self
    def __neg__(self):
        return Tensor(-self.data)

    def numpy(self):
        return self.data

    def matmul(self, other):
        func = Matmul()
        data = func(self, other)
        out = Tensor(data, func=func)
        return out
    
    def sum(self):
        func = Sum()
        out = Tensor(func(self), func=func)
        return out

    def backward(self):
        if self.grad is None and self.data.size == 1:
            self.grad = 1.0
        if self.func is None: return
        # Initialize inputs grads
        for t in self.func.inputs:
            if t.grad is None:
                t.grad = np.zeros_like(t.data)
        # update inputs grads
        grads = self.func.backward(self.grad)
        grads = grads if isinstance(grads, tuple) else (grads, )
        for t, g in zip(self.func.inputs, grads):
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
        return self.forward(*args, **kwargs)

class Add(Function):
    def forward(self, x: Tensor, y: Tensor):
        self.inputs = (x, y)
        return x.data + y.data

    def backward(self, out_grad: Tensor):
        return out_grad, out_grad

class Mul(Function):
    def forward(self, x: Tensor, y: Tensor):
        self.inputs = (x, y)
        return x.data * y.data

    def backward(self, out_grad: Tensor):
        x, y = self.inputs
        x_grad = out_grad * y.data
        y_grad = out_grad * x.data
        return x_grad, y_grad

class Matmul(Function):
    def forward(self, x: Tensor, y: Tensor):
        # x (2, 3)
        # y (3, 4)
        # out (2, 4)
        self.inputs = (x, y)
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

class Sum(Function):
    def forward(self, x):
        self.inputs = (x, )
        return x.data.sum()

    def backward(self, out_grad):
        return out_grad


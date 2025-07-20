from typing import Self
import numpy as np

class Tensor:
    def __init__(self, data, src=None):
        self.data = data
        self.src = src
        self.grad = None
        self._backward = lambda x: ()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other: Self|int|float):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        out = Tensor(self.data + other.data, src=(self, other))
        # TODO: properly handle broadcast case
        def backward(out_grad):
            self.grad += out_grad
            other.grad += out_grad
        out._backward = backward
        return out

    def __radd__(self, other: Self|int|float):
        return self + other

    def __mul__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        out = Tensor(self.data * other.data, src=(self, other))
        def backward(out_grad):
            self.grad += out_grad * other.data
            other.grad += out_grad * self.data
        out._backward = backward
        return out

    def __rmul__(self, other: Self|int|float):
        return self * other

    # self - other
    def __sub__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Tensor(other)
        return Tensor(self.data - other.data, src=(self, other))

    # other - self
    def __rsub__(self, other: Self):
        return (-self) + other

    # -self
    def __neg__(self):
        return Tensor(-self.data, src=(self, ))

    def numpy(self):
        return self.data

    def matmul(self, other):
        # self (2, 3)
        # other (3, 4)
        # out (2, 4)
        out = Tensor(self.data.dot(other.data), src=(self, other))
        def backward(out_grad):
            # self (2, 3)
            # other (3, 4)
            # out_grad (2, 4)
            self.grad += out_grad.dot(other.data.transpose())
            other.grad += self.data.transpose().dot(out_grad)
        out._backward = backward
        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), src=(self, ))
        def backward(out_grad):
            self.grad += out_grad
        out._backward = backward
        return out

    def backward(self):
        if self.grad is None and self.data.size == 1:
            self.grad = 1.0
        if self.src is None: return
        # Initialize src grads
        for t in self.src:
            if t.grad is None:
                t.grad = np.zeros_like(t.data)
        # update src grads
        self._backward(self.grad)
        # recursively backward()
        for t in self.src:
            t.backward()

from typing import Self
import numpy as np


def register(F, name):
    def call(self, *args):
        f = F()  # Create fresh instance for each call
        args = (Tensor(x) if isinstance(x, int | float) else x for x in args)
        return Tensor(f(self, *args), f=f, is_leaf=False, requires_grad=f.requires_grad())

    setattr(Tensor, name, call)


class Tensor:
    def __init__(self, data, is_leaf=True, requires_grad=False, f=None):
        self.data = data
        self.f = f
        self.grad = None
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad
        self._backward = lambda x: ()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __radd__(self, other: Self | int | float):
        return self + other

    def __rmul__(self, other: Self | int | float):
        return self * other

    # self - other
    def __sub__(self, other: Self):
        return self + (-other)

    # other - self
    def __rsub__(self, other: Self):
        return (-self) + other

    @property
    def shape(self):
        return self.data.shape

    def detach(self):
        return Tensor(self.data)

    def numpy(self):
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return self.data

    def backward(self, outgrad=None):
        assert (outgrad is not None or self.data.size == 1)
        if self.f is None:
            return
        # Compute gradients
        if outgrad is None:
            grads = self.f.backward(1.0)
        else:
            grads = self.f.backward(outgrad)
        grads = grads if isinstance(grads, tuple) else (grads,)
        # Single loop: initialize, update grads, and recurse
        for t, g in zip(self.f.inputs, grads):
            grad = Tensor(np.zeros_like(t.data))
            grad += Tensor(g)
            if t.requires_grad and t.is_leaf:
                # Update gradient
                if t.grad is None:
                    t.grad = grad
                else:
                    t.grad.data += grad
            # Recurse
            t.backward(t.grad.data if t.requires_grad and t.is_leaf else grad.data)


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


register(Add, "__add__")


class Neg(Function):
    def forward(self, x: Tensor):
        return -x.data

    def backward(self, out_grad: Tensor):
        return -out_grad


register(Neg, "__neg__")


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def backward(self, out_grad: Tensor):
        x, y = self.inputs
        x_grad = out_grad * y.data
        y_grad = out_grad * x.data
        return x_grad, y_grad


register(Mul, "__mul__")


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


register(Matmul, "matmul")


class Sum(Function):
    def forward(self, x):
        return x.data.sum()

    def backward(self, out_grad):
        return out_grad


register(Sum, "sum")

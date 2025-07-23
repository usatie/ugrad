import ugrad
import torch
import numpy as np

"""
ComparableTensor enables easy comparison between a PyTorch tensor and a ugrad Tensor. Assuming the same operations are supported by both, it allows you to verify that the results are consistent.
"""


class ComparableTensor:
    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            a, b = args
            if isinstance(a, torch.Tensor):
                self.a = a
                self.b = b
                return
        self.a = torch.tensor(*args, **kwargs)
        self.b = ugrad.Tensor(*args, **kwargs)

    @property
    def torch(self):
        return self.a

    @property
    def ugrad(self):
        return self.b

    def __getattr__(self, name):
        def method_forwarder(*args, **kwargs):
            a_args = []
            b_args = []
            for arg in args:
                if isinstance(arg, ComparableTensor):
                    a_args.append(arg.a)
                    b_args.append(arg.b)
                else:
                    a_args.append(arg)
                    b_args.append(arg)
            result_a = getattr(self.a, name)(*a_args, **kwargs)
            result_b = getattr(self.b, name)(*b_args, **kwargs)
            if result_a is None and result_b is None:
                return None
            return ComparableTensor(result_a, result_b)

        return method_forwarder

    def __repr__(self):
        grad_a = (
            self.a.grad.numpy()
            if self.a.grad is not None and self.a.requires_grad and self.a.is_leaf
            else self.a.grad
        )
        grad_b = (
            self.b.grad.numpy()
            if self.b.grad is not None and self.b.requires_grad and self.a.is_leaf
            else self.b.grad
        )
        return f"<torch {self.a.detach().numpy()}, grad={grad_a}, requires_grad={self.a.requires_grad} is_leaf={self.a.is_leaf}>\n<ugrad {self.b.detach().numpy()}, grad={grad_b}, requires_grad={self.b.requires_grad} is_leaf={self.b.is_leaf}>"

    @property
    def grad(self):
        grad = ComparableTensor([])
        grad.a = self.a.grad
        grad.b = self.b.grad
        return grad

    @property
    def shape(self):
        assert self.ugrad.shape == self.torch.shape
        return self.ugrad.shape

    @property
    def data(self):
        self.assert_data_equal()
        return self.ugrad.data

    def assert_all(self):
        assert self.a.shape == self.b.shape
        # assert(self.a.dtype == self.b.dtype)
        assert self.a.requires_grad == self.b.requires_grad
        assert self.a.is_leaf == self.b.is_leaf
        self.assert_data_equal()
        self.assert_grad_equal()

    def assert_data_equal(self):
        np.testing.assert_allclose(
            self.a.detach().numpy(), self.b.detach().numpy(), atol=1e-6
        )

    def assert_grad_equal(self):
        if self.a.requires_grad and self.a.is_leaf:
            np.testing.assert_allclose(
                self.a.grad.numpy(), self.b.grad.numpy(), atol=1e-6
            )
        else:
            assert self.b.grad is None


def register(name):
    setattr(
        ComparableTensor,
        name,
        lambda self, *args, **kwargs: self.__getattr__(name)(*args, **kwargs),
    )


register("__sub__")
register("__add__")
register("__mul__")
register("__neg__")
register("__pow__")
register("__rmul__")

"""
ComparableSGD is a wrapper around both PyTorch and ugrad SGD optimizers.
"""


class ComparableSGD:
    def __init__(self, params, *args, **kwargs):
        self.a = torch.optim.SGD([p.torch for p in params], *args, **kwargs)
        self.b = ugrad.optim.SGD([p.ugrad for p in params], *args, **kwargs)

    def zero_grad(self):
        self.a.zero_grad()
        self.b.zero_grad()

    def step(self):
        self.a.step()
        self.b.step()

import ugrad
import torch
import numpy as np

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
        grad_a = self.a.grad.numpy() if self.a.requires_grad and self.a.is_leaf else self.a.grad
        grad_b = self.b.grad.numpy() if self.b.requires_grad and self.a.is_leaf else self.b.grad
        return f"<torch {self.a.detach().numpy()}, grad={grad_a}, requires_grad={self.a.requires_grad}>\n<ugrad {self.b.detach().numpy()}, grad={grad_b}, requires_grad={self.b.requires_grad}>"

    
    def assert_all(self):
        assert(self.a.shape == self.b.shape)
        #assert(self.a.dtype == self.b.dtype)
        assert(self.a.requires_grad == self.b.requires_grad)
        assert(self.a.is_leaf == self.b.is_leaf)
        self.assert_data_equal()
        self.assert_grad_equal()

    def assert_data_equal(self):
        np.testing.assert_allclose(self.a.detach().numpy(), self.b.detach().numpy())
    def assert_grad_equal(self):
        if self.a.requires_grad and self.a.is_leaf:
            np.testing.assert_allclose(self.a.grad.numpy(), self.b.grad.numpy())
        else:
            assert(self.b.grad is None)

def register(name):
    setattr(ComparableTensor, name, lambda self, *args, **kwargs: self.__getattr__(name)(*args, **kwargs))

register('__sub__')
register('__add__')
register('__mul__')
register('__neg__')

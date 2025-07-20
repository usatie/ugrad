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
        grad_a = self.a.grad.numpy() if isinstance(self.a.grad, torch.Tensor) else self.a.grad
        grad_b = self.b.grad.numpy() if isinstance(self.b.grad, ugrad.Tensor) else self.b.grad
        return f"<torch {self.a.detach().numpy()}, grad={grad_a}>\n<ugrad {self.b.detach().numpy()}, grad={grad_b}>"

    
    def assert_data_equal(self):
        np.testing.assert_allclose(self.a.detach().numpy(), self.b.detach().numpy())
    def assert_grad_equal(self):
        if self.a.grad is None:
            assert(self.a.grad == self.b.grad)
        else:
            np.testing.assert_allclose(self.a.grad.numpy(), self.b.grad.numpy())

def register(name):
    setattr(ComparableTensor, name, lambda self, *args, **kwargs: self.__getattr__(name)(*args, **kwargs))

register('__sub__')
register('__add__')
register('__mul__')
register('__neg__')

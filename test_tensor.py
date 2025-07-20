from ugrad import Tensor
import numpy as np
import torch

def test_add():
    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))
    c = a + b
    assert np.all(np.array([5, 7, 9]) == c.data)
    assert np.all((a + 4).data == np.array([5, 6, 7]))
    assert np.all((4 + a).data == np.array([5, 6, 7]))

def test_sub():
    a = Tensor(np.array([5, 6, 7]))
    b = Tensor(np.array([3, 2, 1]))
    c = a - b
    assert np.all(np.array([2, 4, 6]) == c.data)
    assert np.all((a - 2).data == np.array([3, 4, 5]))
    assert np.all((2 - a).data == np.array([-3, -4, -5]))

def test_mul():
    a = Tensor(np.array([3, 4, 5]))
    b = Tensor(np.array([2, 3, 4]))
    c = a * b
    assert np.all(np.array([6, 12, 20]) == c.data)
    assert np.all((a * 2).data == np.array([6, 8, 10]))
    assert np.all((2 * a).data == np.array([6, 8, 10]))

def test_neg():
    a = Tensor(np.array([1, -2, 3]))
    b = -a
    assert np.all(np.array([-1, 2, -3]) == b.data)

def test_matmul():
    w = np.random.randn(1, 3)
    x = np.random.randn(3, 3)
    b = np.random.randn(1, 3)

    uw = Tensor(w)
    ux = Tensor(x)
    ub = Tensor(b)
    uy = uw.matmul(ux) + ub

    tw = torch.tensor(w, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    ty = tw.matmul(tx) + tb
    assert np.allclose(uy.numpy(), ty.detach().numpy())

    ty.sum().backward()
    out = uy.sum()
    out.grad = 1.0
    out.backward()
    assert np.allclose(tx.grad.numpy(), ux.grad)
    assert np.allclose(tb.grad.numpy(), ub.grad)
    assert np.allclose(tw.grad.numpy(), uw.grad)
    

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


def test_grad():
    w = np.random.randn(1, 3)
    s = np.random.randn(1, 3)
    m = np.random.randn(1, 3)
    n = np.random.randn(1, 3)
    x = np.random.randn(3, 3)
    b = np.random.randn(1, 3)

    # ugrad
    uw = Tensor(w, requires_grad=True)
    us = Tensor(s, requires_grad=True)
    um = Tensor(m, requires_grad=True)
    un = Tensor(n, requires_grad=True)
    ux = Tensor(x, requires_grad=True)
    ub = Tensor(b, requires_grad=True)

    uW = (uw - us) * um + (-un)
    uwx = uW.matmul(ux)
    uy = uwx + ub
    uout = uy.sum()
    uout.backward()

    # pytorch
    tw = torch.tensor(w, requires_grad=True)
    ts = torch.tensor(s, requires_grad=True)
    tm = torch.tensor(m, requires_grad=True)
    tn = torch.tensor(n, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)

    tW = (tw - ts) * tm + (-tn)
    ty = tW.matmul(tx) + tb
    tout = ty.sum()
    tout.backward()

    assert np.allclose(ty.detach().numpy(), uy.detach().numpy())
    assert np.allclose(tout.detach().numpy(), uout.detach().numpy())
    assert np.allclose(tn.grad.numpy(), un.grad.numpy())
    assert np.allclose(tm.grad.numpy(), um.grad.numpy())
    assert np.allclose(ts.grad.numpy(), us.grad.numpy())
    assert np.allclose(tb.grad.numpy(), ub.grad.numpy())
    assert np.allclose(tx.grad.numpy(), ux.grad.numpy())
    assert np.allclose(tw.grad.numpy(), uw.grad.numpy())

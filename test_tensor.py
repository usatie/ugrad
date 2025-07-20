from ugrad import Tensor
import numpy as np
from helper import ComparableTensor


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


np_w = np.random.randn(1, 3)
np_s = np.random.randn(1, 3)
np_m = np.random.randn(1, 3)
np_n = np.random.randn(1, 3)
np_x = np.random.randn(3, 3)
np_b = np.random.randn(1, 3)

def test_grad():
    w = ComparableTensor(np_w, requires_grad=True)
    s = ComparableTensor(np_s, requires_grad=True)
    m = ComparableTensor(np_m, requires_grad=True)
    n = ComparableTensor(np_n, requires_grad=True)
    x = ComparableTensor(np_x, requires_grad=True)
    b = ComparableTensor(np_b, requires_grad=True)

    W = (w - s) * m + (-n)
    wx = W.matmul(x)
    y = wx + b
    out = y.sum()
    out.backward()

    w.assert_all()
    s.assert_all()
    m.assert_all()
    n.assert_all()
    x.assert_all()
    b.assert_all()
    W.assert_all()
    wx.assert_all()
    y.assert_all()
    out.assert_all()

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


def test_t():
    a = ComparableTensor(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True
    )
    b = a.t()
    assert np.all(b.ugrad.data == np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32))
    c = b * 2
    loss = c.sum()
    loss.backward()

    a.assert_all()
    b.assert_all()
    c.assert_all()


def test_relu():
    a = ComparableTensor(
        np.arange(0, 6, dtype=np.double).reshape(2, 3), requires_grad=True
    )
    b = ComparableTensor(
        np.arange(-4, 2, dtype=np.double).reshape(3, 2), requires_grad=True
    )
    c = a.matmul(b).relu()
    d = ComparableTensor(np.random.randn(2, 1), requires_grad=True)
    e = c.matmul(d).relu()
    f = e.sum()
    f.backward()

    f.assert_all()
    e.assert_all()
    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()


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


class LinearLayer:
    def __init__(self, size_in, size_out):
        self.W = ComparableTensor(
            np.random.randn(size_out, size_in), requires_grad=True
        )
        self.b = ComparableTensor(np.random.randn(size_out), requires_grad=True)

    # x : (bs, size_in)
    # out : (bs, size_out)
    def __call__(self, x):
        z = x.matmul(self.W.t()) + self.b
        return z.relu()


def test_mlp():
    x = ComparableTensor(np.random.randn(32, 2))
    np_y = np.random.randint(0, 2, 32).reshape(32, 1)
    y = ComparableTensor(np_y)
    l1 = LinearLayer(2, 4)
    l2 = LinearLayer(4, 8)
    l3 = LinearLayer(8, 1)
    out = l3(l2(l1(x)))
    mse = (y - out) ** 2
    loss = mse.sum()

    loss.backward()
    loss.assert_all()
    mse.assert_all()
    out.assert_all()
    l3.W.assert_all()
    l3.b.assert_all()
    l2.W.assert_all()
    l2.b.assert_all()
    l1.W.assert_all()
    l1.b.assert_all()
    y.assert_all()
    x.assert_all()

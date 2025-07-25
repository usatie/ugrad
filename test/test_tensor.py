from ugrad import Tensor
import numpy as np
import torch
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


def test_softmax():
    a = ComparableTensor(
        np.arange(0, 6, dtype=np.double).reshape(2, 3), requires_grad=True
    )
    b = ComparableTensor(
        np.arange(-4, 2, dtype=np.double).reshape(3, 2), requires_grad=True
    )
    c = a.matmul(b).softmax(1)
    d = ComparableTensor(np.random.randn(2, 1), requires_grad=True)
    e = c.matmul(d).softmax(1)
    f = e.sum()
    f.backward()

    f.assert_all()
    e.assert_all()
    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()

def test_log():
    a = ComparableTensor(
        1 + np.random.randn(2, 3), requires_grad=True
    )
    b = ComparableTensor(
        1 + np.random.randn(3, 2), requires_grad=True
    )
    c = a.matmul(b).log()
    d = c.sum()
    d.backward()

    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()

def test_conv2d():
    N, in_channel, out_channel, W, H = 4, 2, 3, 3, 3
    ks = 2
    x = ComparableTensor(np.random.randn(N, in_channel, W, H), requires_grad=True)
    filters = ComparableTensor(
        np.random.randn(out_channel, in_channel, ks, ks), requires_grad=True
    )
    out = x.conv2d(filters)
    m = ComparableTensor(np.random.randn(N, out_channel, W - ks + 1, W - ks + 1))
    (out * m).sum().backward()
    out.assert_all()
    x.assert_all()
    filters.assert_all()


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
    def __init__(self, size_in, size_out, activation=True):
        self.activation = activation
        self.W = ComparableTensor(
            np.random.randn(size_out, size_in), requires_grad=True
        )
        self.b = ComparableTensor(np.random.randn(size_out), requires_grad=True)

    # x : (bs, size_in)
    # out : (bs, size_out)
    def __call__(self, x):
        z = x.matmul(self.W.t()) + self.b
        if self.activation:
            return z.relu()
        else:
            return z

    def zero_grad(self):
        self.W.torch.grad = None
        self.b.torch.grad = None
        self.W.ugrad.grad = None
        self.b.ugrad.grad = None


def test_mlp():
    x = ComparableTensor(np.random.randn(32, 2))
    np_y = np.random.randint(0, 2, 32).reshape(32, 1)
    y = ComparableTensor(np_y)
    l1 = LinearLayer(2, 4)
    l2 = LinearLayer(4, 8)
    l3 = LinearLayer(8, 1, activation=False)

    for i in range(1000):
        l1.zero_grad()
        l2.zero_grad()
        l3.zero_grad()
        out = l3(l2(l1(x)))
        mse = (y - out) ** 2
        loss = mse.sum()
        loss.backward()
        if i % 10 == 0:
            # print(f"i = {i}")
            # print(loss)
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
        # This hack is needed because we don't have torch.no_grad() like context manager yet
        l1.W.torch.data -= 0.001 * l1.W.torch.grad.data
        l1.W.ugrad.data -= 0.001 * l1.W.ugrad.grad.data
        l1.b.torch.data -= 0.001 * l1.b.torch.grad.data
        l1.b.ugrad.data -= 0.001 * l1.b.ugrad.grad.data
        l2.W.torch.data -= 0.001 * l2.W.torch.grad.data
        l2.W.ugrad.data -= 0.001 * l2.W.ugrad.grad.data
        l2.b.torch.data -= 0.001 * l2.b.torch.grad.data
        l2.b.ugrad.data -= 0.001 * l2.b.ugrad.grad.data
        l3.W.torch.data -= 0.001 * l3.W.torch.grad.data
        l3.W.ugrad.data -= 0.001 * l3.W.ugrad.grad.data
        l3.b.torch.data -= 0.001 * l3.b.torch.grad.data
        l3.b.ugrad.data -= 0.001 * l3.b.ugrad.grad.data
    assert loss.ugrad.data < 10.0

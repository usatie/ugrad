import numpy as np
import torch
import pytest


from ugrad import Tensor
from helper import ComparableTensor


def test_broadcast_add():
    a = ComparableTensor(1.0)
    b = ComparableTensor(2.0)
    c = ComparableTensor(np.ones((2, 3)))
    ab = a + b
    abc = ab + c
    ab.assert_all()
    abc.assert_all()


def test_add():
    # Integer addition
    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))
    c = a + b
    d = c + 10
    assert np.all(np.array([5, 7, 9]) == c.npdata)
    assert np.all((a + 4).npdata == np.array([5, 6, 7]))
    assert np.all((4 + a).npdata == np.array([5, 6, 7]))
    assert np.all(d.npdata == np.array([15, 17, 19]))

    # Float addition
    a = Tensor(np.array([1.5, 2.5, 3.5]))
    b = Tensor(np.array([4.2, 5.3, 6.4]))
    c = a + b
    assert np.all(np.array([5.7, 7.8, 9.9]) == c.npdata)
    assert np.all((a + 4.5).npdata == np.array([6.0, 7.0, 8.0]))
    assert np.all((4.5 + a).npdata == np.array([6.0, 7.0, 8.0]))

    # 2D addition
    a = Tensor(np.array([[1, 2], [3, 4]]))
    b = Tensor(np.array([[5, 6], [7, 8]]))
    c = a + b
    assert np.all(np.array([[6, 8], [10, 12]]) == c.npdata)

    # 2D addition (float)
    a = Tensor(np.array([[1.5, 2.5], [3.5, 4.5]]))
    b = Tensor(np.array([[5.1, 6.2], [7.3, 8.4]]))
    c = a + b
    assert np.all(np.array([[6.6, 8.7], [10.8, 12.9]]) == c.npdata)

    # Broadcasting
    a = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    b = Tensor(np.array([10, 20, 30]))
    c = Tensor(np.array([[10], [20]]))
    assert np.all(np.array([[11, 22, 33], [14, 25, 36]]) == (a + b).npdata)
    assert np.all(np.array([[11, 12, 13], [24, 25, 26]]) == (a + c).npdata)


def test_sub():
    a = Tensor(np.array([5, 6, 7]))
    b = Tensor(np.array([3, 2, 1]))
    c = a - b
    assert np.all(np.array([2, 4, 6]) == c.npdata)
    assert np.all((a - 2).npdata == np.array([3, 4, 5]))
    assert np.all((2 - a).npdata == np.array([-3, -4, -5]))


def test_mul():
    a = Tensor(np.array([3, 4, 5]))
    b = Tensor(np.array([2, 3, 4]))
    c = a * b
    assert np.all(np.array([6, 12, 20]) == c.npdata)
    assert np.all((a * 2).npdata == np.array([6, 8, 10]))
    assert np.all((2 * a).npdata == np.array([6, 8, 10]))


def test_div():
    a = ComparableTensor(np.array([6.0, 8.0, 10.0]), requires_grad=True)
    b = ComparableTensor(2.0, requires_grad=True)
    c = a / b
    d = c.sum()
    d.backward()
    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_sumdiv():
    a = ComparableTensor(
        np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]), requires_grad=True
    )
    b = a.sum(1, keepdim=True)
    c = a / b
    d = c.sum()
    d.backward()
    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()


"""
If a tensor needs to be updated twice in the backward pass, it will fail?
sum(keepdim=True)
"""


def test():
    a = ComparableTensor(
        np.array([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]), requires_grad=True
    )
    b = a.sum()
    b.requires_grad = True
    b.is_leaf = True
    c = a * b
    d = c.sum()
    d.backward()
    d.assert_all()
    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_neg():
    a = Tensor(np.array([1, -2, 3]))
    b = -a
    assert np.all(np.array([-1, 2, -3]) == b.npdata)


def test_t():
    a = ComparableTensor(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True
    )
    b = a.t()
    assert np.all(
        b.ugrad.npdata == np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    )
    c = b * 2
    loss = c.sum()
    loss.backward()
    loss.assert_all()

    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_lt():
    a = ComparableTensor(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True
    )
    b = ComparableTensor(
        np.array([[2, 2, 3], [4, 5, 7]], dtype=np.float32), requires_grad=True
    )
    (a < b).assert_all()
    (a <= b).assert_all()
    (a > b).assert_all()
    (a < 2).assert_all()
    (a <= 2).assert_all()
    (a > 2).assert_all()


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


def test_cosine():
    a = ComparableTensor(
        np.arange(0, 6, dtype=np.double).reshape(2, 3), requires_grad=True
    )
    b = a.cos()
    c = b.sum()
    c.backward()
    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_softmax():
    a = ComparableTensor(
        np.arange(0, 6, dtype=np.double).reshape(2, 3), requires_grad=True
    )
    b = a.softmax(1)
    c = b.sum()
    c.backward()

    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_log():
    a = ComparableTensor(1 + np.random.randn(2, 3), requires_grad=True)
    b = a.log()
    c = b.sum()
    c.backward()

    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_log_softmax():
    a = ComparableTensor(np.random.randn(2, 3), requires_grad=True)
    b = a.log_softmax(1)
    c = b.sum()
    c.backward()

    c.assert_all()
    b.assert_all()
    a.assert_all()


def test_square():
    a = ComparableTensor(np.random.randn(2, 3), requires_grad=True)
    a.square().assert_all()


def test_sqrt():
    a = ComparableTensor(np.random.randn(2, 3), requires_grad=True)
    a.sqrt().assert_all()


def test_mean():
    """
    a = ComparableTensor(np.random.randn(2, 3), requires_grad=True)
    b = a.mean(1)
    c = b.sum()
    c.backward()

    c.assert_all()
    b.assert_all()
    a.assert_all()
    """

    d = ComparableTensor(np.random.randn(2, 3), requires_grad=True)
    e = d.mean()
    f = e.sum()
    f.backward()
    d.assert_all()
    return

    e = ComparableTensor(np.random.randn(2, 3, 4), requires_grad=True)
    e.mean(0).sum().backward()
    e.assert_all()

    f = ComparableTensor(np.random.randn(2, 3, 4), requires_grad=True)
    f.mean(2).sum().backward()
    f.assert_all()

    x = ComparableTensor(np.random.randn(2, 3, 4), requires_grad=True)
    x.mean().assert_all()
    x.mean(0, keepdim=True).assert_all()
    x.mean(1, keepdim=True).assert_all()
    x.mean(2, keepdim=True).assert_all()
    x.mean(0, keepdim=False).assert_all()
    x.mean(1, keepdim=False).assert_all()
    x.mean(2, keepdim=False).assert_all()


def test_sum():
    a = ComparableTensor(np.random.randn(2, 3, 4), requires_grad=True)

    a.sum().assert_all()
    a.sum(0).assert_all()
    a.sum(1).assert_all()
    a.sum(2).assert_all()


def test_std():
    a = ComparableTensor(
        np.arange(0, 24, dtype=np.float64).reshape((2, 12)), requires_grad=True
    )
    a.std(1, correction=0).assert_all()
    a.std(1, correction=1).assert_all()
    a.std(correction=0).assert_all()
    a.std(correction=1).assert_all()
    a.std().assert_all()
    a.std(1).sum().backward()
    a.assert_all()


def test_linear():
    N, in_channel, out_channel = 4, 6, 3
    x = ComparableTensor(np.random.randn(N, in_channel), requires_grad=False)
    W = ComparableTensor(np.random.randn(out_channel, in_channel), requires_grad=True)
    b = ComparableTensor(np.random.randn(N, out_channel), requires_grad=True)
    out = x.linear(W, b)
    out.assert_all()
    m = ComparableTensor(np.random.randn(N, out_channel))
    (out * m).sum().backward()
    W.assert_all()
    b.assert_all()


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


def test_zeros():
    ComparableTensor.zeros(42).assert_all()
    ComparableTensor.zeros((42,)).assert_all()
    ComparableTensor.zeros((2, 3, 4)).assert_all()


def test_zeros_like():
    a = ComparableTensor(np.random.randn(42))
    b = ComparableTensor(np.random.randn(2, 3, 4))
    ComparableTensor.zeros_like(a).assert_all()
    ComparableTensor.zeros_like(b).assert_all()


@pytest.mark.slow
def test_randn():
    a = Tensor.randn(10, 20, 30, 40)
    assert a.shape == (10, 20, 30, 40)
    assert abs(a.mean().npdata) < 0.01, "Mean should be close to 0"
    assert abs(a.std().npdata - 1.0) < 0.01, "Std should be close to 1"


@pytest.mark.slow
def test_normal():
    a = Tensor.normal(10, 20, 30, 40, mean=42.0, std=3.0)
    assert a.shape == (10, 20, 30, 40)
    assert abs(a.mean().npdata - 42.0) < 0.1, "Mean should be close to 42.0"
    assert abs(a.std().npdata - 3.0) < 0.1, "Std should be close to 3.0"


@pytest.mark.slow
def test_uniform():
    a = Tensor.uniform(10, 20, 30, 40, low=0.0, high=1.0)
    assert a.shape == (10, 20, 30, 40)
    assert np.all(a.npdata >= 0.0) and np.all(
        a.npdata <= 1.0
    ), "Values should be in [0.0, 1.0]"
    # How can I test the uniformity of the distribution?
    assert abs(a.mean().npdata - 0.5) < 0.1, "Mean should be close to 0.5"
    assert abs(a.std().npdata - 0.29) < 0.1, "Std should be close to 0.29"
    assert (
        abs((a.npdata < 0.1).sum() - a.npdata.size * 0.1) < a.npdata.size * 0.01
    ), "Approximately 10% of values should be < 0.1"
    assert (
        abs((a.npdata < 0.2).sum() - a.npdata.size * 0.2) < a.npdata.size * 0.01
    ), "Approximately 20% of values should be < 0.2"
    assert (
        abs((a.npdata < 0.3).sum() - a.npdata.size * 0.3) < a.npdata.size * 0.01
    ), "Approximately 30% of values should be < 0.3"
    assert (
        abs((a.npdata < 0.4).sum() - a.npdata.size * 0.4) < a.npdata.size * 0.01
    ), "Approximately 40% of values should be < 0.4"
    assert (
        abs((a.npdata < 0.5).sum() - a.npdata.size * 0.5) < a.npdata.size * 0.01
    ), "Approximately 50% of values should be < 0.5"
    assert (
        abs((a.npdata < 0.6).sum() - a.npdata.size * 0.6) < a.npdata.size * 0.01
    ), "Approximately 60% of values should be < 0.6"
    assert (
        abs((a.npdata < 0.7).sum() - a.npdata.size * 0.7) < a.npdata.size * 0.01
    ), "Approximately 70% of values should be < 0.7"
    assert (
        abs((a.npdata < 0.8).sum() - a.npdata.size * 0.8) < a.npdata.size * 0.01
    ), "Approximately 80% of values should be < 0.8"
    assert (
        abs((a.npdata < 0.9).sum() - a.npdata.size * 0.9) < a.npdata.size * 0.01
    ), "Approximately 90% of values should be < 0.9"


def test_xavier_uniform():
    a = Tensor.xavier_uniform(10, 20, 3, 3)
    assert a.shape == (10, 20, 3, 3)
    assert abs(a.mean().npdata) < 0.01, "Mean should be close to 0"
    fan_out = 10 * 3 * 3
    fan_in = 20 * 3 * 3
    bound = np.sqrt(6 / (fan_in + fan_out))
    assert abs(a.npdata.min() + bound) < 0.01, "Min should be close to -bound"
    assert abs(a.npdata.max() - bound) < 0.01, "Max should be close to bound"


def _test_kaiming_uniform(nonlinearity, gain):
    if nonlinearity is None:
        a = Tensor.kaiming_uniform(10, 20, 3, 3)
    else:
        a = Tensor.kaiming_uniform(10, 20, 3, 3, nonlinearity=nonlinearity)
    assert a.shape == (10, 20, 3, 3)
    assert abs(a.mean().npdata) < 0.01, "Mean should be close to 0"
    fan_in = 20 * 3 * 3
    bound = gain * np.sqrt(3 / fan_in)
    assert abs(a.npdata.min() + bound) < 0.01, "Min should be close to -bound"
    assert abs(a.npdata.max() - bound) < 0.01, "Max should be close to bound"


def test_kaiming_uniform_default():
    _test_kaiming_uniform(None, gain=np.sqrt(2.0))


def test_kaiming_uniform_relu():
    _test_kaiming_uniform("relu", gain=np.sqrt(2.0))


def test_kaiming_uniform_sigmoid():
    _test_kaiming_uniform("sigmoid", gain=1.0)


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


@pytest.mark.slow
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
        l1.W.ugrad.assign(l1.W.ugrad - 0.001 * l1.W.ugrad.grad)
        l1.b.torch.data -= 0.001 * l1.b.torch.grad.data
        l1.b.ugrad.assign(l1.b.ugrad - 0.001 * l1.b.ugrad.grad)
        l2.W.torch.data -= 0.001 * l2.W.torch.grad.data
        l2.W.ugrad.assign(l2.W.ugrad - 0.001 * l2.W.ugrad.grad)
        l2.b.torch.data -= 0.001 * l2.b.torch.grad.data
        l2.b.ugrad.assign(l2.b.ugrad - 0.001 * l2.b.ugrad.grad)
        l3.W.torch.data -= 0.001 * l3.W.torch.grad.data
        l3.W.ugrad.assign(l3.W.ugrad - 0.001 * l3.W.ugrad.grad)
        l3.b.torch.data -= 0.001 * l3.b.torch.grad.data
        l3.b.ugrad.assign(l3.b.ugrad - 0.001 * l3.b.ugrad.grad)
    assert loss.ugrad.npdata < 10.0


def test_broadcast():
    x = ComparableTensor(np.random.randn(2, 3))
    one = ComparableTensor(1.0)
    y = x + 1
    y.assert_all()

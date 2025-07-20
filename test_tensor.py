from ugrad import Tensor
import numpy as np

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

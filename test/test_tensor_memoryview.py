import numpy as np
from math import prod

class Tensor:
    def __init__(
            self,
            mv: memoryview,
            shape: tuple[int] = None,
            strides: tuple[int] = None,
            offset: int = 0
            ):
        self.data = mv.cast('B').cast(mv.format)
        if shape is None:
            self._shape = mv.shape
        else:
            # not necessarily match, especially when sliced
            #assert prod(shape) == prod(mv.shape)
            self._shape = shape
        if strides is None:
            self._strides = mv.strides
        else:
            # TODO: Check validity of strides
            assert len(self._shape) == len(strides)
            self._strides = strides
        self.offset = offset

    def __repr__(self) -> str:
        return f"Tensor(data={self.tolist()}, shape={self.shape}, strides={self.strides}, offset={self.offset})"

    def tolist(self) -> list:
        l = []
        from itertools import product
        for idx in product(*(range(sh) for sh in self.shape)):
            l.append(self[idx])
        return l

    @property
    def shape(self) -> tuple[int]:
        return self._shape

    @property
    def strides(self) -> tuple[int]:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def reshape(self, *shape: int):
        # TODO: Check contiguous (not always reshape-able)
        assert prod(self.shape) == prod(shape)
        total = self.data.nbytes
        new_strides = tuple((total := total // sh) for sh in shape)
        return Tensor(self.data, shape, new_strides, self.offset)

    def transpose(self):
        # Let's transpose first and second dim
        i, j = 0, 1
        shape, strides = [], []
        for idx, (sh, st) in enumerate(zip(self.shape, self.strides)):
            if idx == i:
                shape.append(self.shape[j])
                strides.append(self.strides[j])
            elif idx == j:
                shape.append(self.shape[i])
                strides.append(self.strides[i])
            else:
                shape.append(sh)
                strides.append(st)

        return Tensor(self.data, tuple(shape), tuple(strides), self.offset)

    def _getindex(self, idx):
        index = (self.offset + sum((sh * i for sh, i in zip(self.strides, idx)))) // self.data.itemsize
        return index

    def __getitem__(self, idx):
        if isinstance(idx, int): idx = (idx, )
        assert type(idx) == tuple
        assert len(idx) <= self.ndim
        if len(idx) == len(self.shape) and all(isinstance(x, int) for x in idx):
            index = self._getindex(idx)
            return self.data[index]
        else:
            # TODO: Support slices
            # Assume idx is a integer
            offset = self.offset
            for i, st, sh in zip(idx, self.strides, self.shape):
                if i < 0 or i >= sh:
                    raise IndexError("Index out of bound")
                offset += i * st
            return Tensor(self.data, self.shape[len(idx):], self.strides[len(idx):], offset)

    @property
    def T(self):
        return self.transpose()

def test_constructor():
    a = np.arange(0, 12).reshape(3, 4)
    b = Tensor(a.data)

    assert b.shape == (3, 4)
    assert b.strides == (32, 8)

def _assert_all(n, t):
    assert n.shape == t.shape
    assert n.strides == t.strides
    from itertools import product
    for idx in product(*(range(sh) for sh in n.shape)):
        assert n[idx] == t[idx]

    if len(n.shape) > 1:
        for idx in range(n.shape[0]):
            _assert_all(n[idx], t[idx])

def _assert_except(n, t, f, exctype):
    import pytest
    with pytest.raises(exctype):
        f(n)
    with pytest.raises(exctype):
        f(t)

def test_getitem():
    # 1D
    a = np.arange(0, 12)
    b = Tensor(a.data)
    _assert_all(a, b)
    assert a[3] == b[3]

    # 2D
    a = np.arange(0, 12).reshape(3, 4)
    b = Tensor(a.data)
    _assert_all(a, b)
    _assert_all(a[1], b[1])
    assert a[0][1] == b[0][1]

    # 3D
    a = np.arange(0, 12).reshape(2, 2, 3)
    b = Tensor(a.data)
    _assert_all(a, b)
    _assert_all(a[1], b[1])
    _assert_all(a[0][1], b[0][1])
    _assert_all(a[0, 1], b[0, 1])
    assert a[0][1][1] == b[0][1][1]

    # Out of bound
    # Test if a and b raises the same error
    _assert_except(a, b, lambda x: x[3, 0, 0], IndexError)
    _assert_except(a, b, lambda x: x[3][0][0], IndexError)
    _assert_except(a, b, lambda x: x[3], IndexError)
    _assert_except(a, b, lambda x: x[0][3], IndexError)
    _assert_except(a, b, lambda x: x[0, 4], IndexError)

def test_reshape():
    a = np.arange(0, 12)
    b = Tensor(a.data)
    _assert_all(a, b)
    _assert_all(a.reshape(3, 4), b.reshape(3, 4))
    _assert_all(a.reshape(2, 3, 2), b.reshape(2, 3, 2))
    _assert_all(a.reshape(3, 4).reshape(12), b.reshape(3, 4).reshape(12))


def test_transpose():
    a = np.arange(0, 12).reshape(3,4)     
    b = Tensor(a.data)
    _assert_all(a, b)
    _assert_all(a.transpose(), b.transpose())
    _assert_all(a.T, b.T)

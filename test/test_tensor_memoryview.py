import numpy as np
from ugrad.shape.shapetracker import ShapeTracker
from ugrad.shape.view import View


class Tensor:
    def __init__(
        self,
        data: np.ndarray | memoryview,
        st: ShapeTracker = None,
    ):
        if isinstance(data, np.ndarray):
            mv = (
                data.data
                if data.flags.c_contiguous
                else np.ascontiguousarray(data).data
            )
        elif isinstance(data, memoryview):
            mv = data
        # To ensure the memoryview is 1D
        self.data = mv.cast("B").cast(mv.format)
        if st is None:
            shape = mv.shape
            strides = tuple(st // mv.itemsize for st in mv.strides)
            st = ShapeTracker((View(shape, strides, 0),))
        self.st = st

    def __repr__(self) -> str:
        def _print(lst, level=0):
            if not isinstance(lst, list):
                return str(lst)
            indent = "  " * level
            if all(not isinstance(i, list) for i in lst):
                return "[" + ", ".join(str(i) for i in lst) + "]"
            else:
                inner = ",\n".join(indent + "  " + _print(i, level + 1) for i in lst)
                return "[\n" + inner + "\n" + indent + "]"

        return f"Tensor(data={_print(self.tolist())}, shape={self.shape})"

    def tolist(self) -> list:
        def build_list(shape: tuple[int, ...], index_prefix: tuple[int, ...] = ()):
            if len(shape) == 1:
                return [self[index_prefix + (i,)] for i in range(shape[0])]
            else:
                return [
                    build_list(shape[1:], index_prefix + (i,)) for i in range(shape[0])
                ]

        return build_list(self.shape)

    @property
    def shape(self) -> tuple[int]:
        return self.st.shape

    @property
    def ndim(self) -> int:
        return self.st.ndim

    def reshape(self, *shape: int):
        return Tensor(self.data, self.st.reshape(*shape))

    def transpose(self, *axes):
        return Tensor(self.data, self.st.transpose(*axes))

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx: int | tuple[int, ...]):
        if isinstance(idx, int):
            idx = (idx,)
        if any(not isinstance(i, int) for i in idx):
            raise TypeError("Only integer indexing is supported")
        if len(idx) > self.ndim:
            raise IndexError("Too many indices for tensor")
        if len(idx) == len(self.shape):
            # Maybe it's dumb to make flat index, if memoryview is multi dimensional, but we ensure data is 1D
            index = self.st.get_index(idx)
            return self.data[index]
        else:
            return Tensor(self.data, self.st.slice(idx))


def test_constructor():
    a = np.arange(0, 12).reshape(3, 4)
    b = Tensor(a.data)

    assert b.shape == (3, 4)


def _assert_all(n, t):
    assert n.shape == t.shape
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
    b = Tensor(a)
    _assert_all(a, b)
    assert a[3] == b[3]

    # 2D
    a = np.arange(0, 12).reshape(3, 4)
    b = Tensor(a)
    _assert_all(a, b)
    _assert_all(a[1], b[1])
    assert a[0][1] == b[0][1]

    # 3D
    a = np.arange(0, 12).reshape(2, 2, 3)
    b = Tensor(a)
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
    b = Tensor(a)
    _assert_all(a, b)
    _assert_all(a.reshape(3, 4), b.reshape(3, 4))
    _assert_all(a.reshape(2, 3, 2), b.reshape(2, 3, 2))
    _assert_all(a.reshape(3, 4).reshape(12), b.reshape(3, 4).reshape(12))

    # Non contiguous reshape
    _assert_all(a.reshape(3, 4).T.reshape(3, 4), b.reshape(3, 4).T.reshape(3, 4))
    a = np.arange(0, 24)
    b = Tensor(a)
    _assert_all(
        a.reshape(2, 3, 4).T.reshape(2, 3, 4), b.reshape(2, 3, 4).T.reshape(2, 3, 4)
    )
    _assert_all(
        a.reshape(2, 3, 4).T.reshape(2, 3, 4).T.reshape(2, 3, 4),
        b.reshape(2, 3, 4).T.reshape(2, 3, 4).T.reshape(2, 3, 4),
    )
    assert len(b.reshape(2, 3, 4).T.st.views) == 1
    assert len(b.reshape(2, 3, 4).T.reshape(2, 3, 4).st.views) == 2
    assert len(b.reshape(2, 3, 4).T.reshape(2, 3, 4).T.reshape(2, 3, 4).st.views) == 3


def test_transpose():
    a = np.arange(0, 12).reshape(3, 4)
    b = Tensor(a)
    _assert_all(a, b)
    _assert_all(a.transpose(), b.transpose())
    _assert_all(a.T, b.T)

    a = np.arange(0, 12).reshape(2, 2, 3)
    b = Tensor(a)
    _assert_all(a.transpose(), b.transpose())
    _assert_all(a.transpose((0, 2, 1)), b.transpose((0, 2, 1)))
    _assert_all(a.transpose(0, 2, 1), b.transpose(0, 2, 1))
    _assert_except(a, b, lambda x: x.transpose((0, 1)), ValueError)


def test_view():
    view = View((2, 3, 4), (12, 4, 1), 3)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                index = view.get_index((i, j, k))
                indices = view.get_indices(index)
                assert (i, j, k) == indices
                assert index == view.get_index(indices)

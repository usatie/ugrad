import numpy as np
from dataclasses import dataclass
from typing import Optional
from math import prod


def strides_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    st = 1
    for sh in reversed(shape):
        strides.append(st)
        st *= sh
    strides = tuple(reversed(strides))
    return canonicalize_strides(strides, shape)


def canonicalize_strides(
    strides: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(0 if sh == 1 else st for sh, st in zip(shape, strides))


@dataclass(frozen=True)
class View:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int

    @staticmethod
    def create(shape: tuple[int, ...]) -> "View":
        return View(shape, strides_for_shape(shape), 0)

    def get_index(self, indices: tuple[int, ...]) -> int:
        index = self.offset + sum((sh * i for sh, i in zip(self.strides, indices)))
        return index

    def get_indices(self, index: int) -> tuple[int, ...]:
        """
        e.g. shape: (2, 3, 4)
        [[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],

         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]
        """
        indices = []
        index -= self.offset
        for sh in reversed(self.shape):
            indices.append(index % sh)
            index = index // sh
        return tuple(reversed(indices))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def contiguous(self) -> bool:
        return self.strides == strides_for_shape(self.shape)

    def reshape(self, *shape: int) -> Optional["View"]:
        # Check contiguous (not always reshape-able)
        if not self.contiguous:
            return None
        assert prod(self.shape) == prod(shape)
        return View(shape, strides_for_shape(shape), self.offset)

    def _transpose(self, axes: Optional[tuple[int, ...]] = None) -> "View":
        if axes is None:
            axes = tuple(range(self.ndim)[::-1])
        if len(axes) != self.ndim:
            raise ValueError("axes don't match array")
        shape = tuple(self.shape[ax] for ax in axes)
        strides = tuple(self.strides[ax] for ax in axes)
        return View(shape, strides, self.offset)

    def transpose(self, *axes: int) -> "View":
        # Let's transpose first and second dim
        if len(axes) == 0:
            return self._transpose(None)
        elif isinstance(axes[0], tuple):
            return self._transpose(axes[0])
        else:
            return self._transpose(axes)

    def slice(self, idx: tuple[int, ...]) -> "View":
        # TODO: Support slice object
        # Assume idx is a integer
        assert all(isinstance(i, int) for i in idx)
        offset = self.offset
        for i, st, sh in zip(idx, self.strides, self.shape):
            if i < 0 or i >= sh:
                raise IndexError("Index out of bound")
            offset += i * st
        return View(self.shape[len(idx) :], self.strides[len(idx) :], offset)


@dataclass(frozen=True)
class ShapeTracker:
    views: tuple[View, ...]

    @property
    def view(self) -> View:
        return self.views[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.view.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def reshape(self, *shape: int) -> "ShapeTracker":
        if (view := self.view.reshape(*shape)) is not None:
            return ShapeTracker(self.views[:-1] + (view,))
        else:
            return ShapeTracker(self.views + (View.create(shape),))

    def slice(self, idx: tuple[int, ...]) -> "ShapeTracker":
        return ShapeTracker(self.views[:-1] + (self.view.slice(idx),))

    def transpose(self, *axes: int) -> "ShapeTracker":
        return ShapeTracker(self.views[:-1] + (self.view.transpose(*axes),))

    def get_index(self, indices: tuple[int, ...]) -> int:
        flat_index = self.views[-1].get_index(indices)
        for view in reversed(self.views[:-1]):
            flat_index = view.get_index(view.get_indices(flat_index))
        return flat_index


class Tensor:
    def __init__(
        self,
        mv: memoryview,
        st: ShapeTracker = None,
    ):
        self.data = mv.cast("B").cast(mv.format)
        if st is None:
            shape = mv.shape
            strides = tuple(st // mv.itemsize for st in mv.strides)
            st = ShapeTracker((View(shape, strides, 0),))
        self.st = st

    def __repr__(self) -> str:
        return f"Tensor(data={self.tolist()}, shape={self.shape})"

    def tolist(self) -> list:
        l = []
        from itertools import product

        for idx in product(*(range(sh) for sh in self.shape)):
            l.append(self[idx])
        return l

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
        assert type(idx) == tuple
        assert len(idx) <= self.ndim
        if len(idx) == len(self.shape) and all(isinstance(x, int) for x in idx):
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

    # Non contiguous reshape
    _assert_all(a.reshape(3, 4).T.reshape(3, 4), b.reshape(3, 4).T.reshape(3, 4))
    a = np.arange(0, 24)
    b = Tensor(a.data)
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
    b = Tensor(a.data)
    _assert_all(a, b)
    _assert_all(a.transpose(), b.transpose())
    _assert_all(a.T, b.T)

    a = np.arange(0, 12).reshape(2, 2, 3)
    b = Tensor(a.data)
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

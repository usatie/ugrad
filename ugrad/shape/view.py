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
    def size(self) -> int:
        return prod(self.shape)

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

from dataclasses import dataclass

from ugrad.shape.view import View


@dataclass(frozen=True)
class ShapeTracker:
    views: tuple[View, ...]

    @staticmethod
    def create(shape: tuple[int, ...]) -> "View":
        return ShapeTracker((View.create(shape),))

    @property
    def size(self) -> int:
        return self.view.size

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

    def get_flat_index(self, indices: tuple[int, ...]) -> int:
        flat_index = self.views[-1].get_index(indices)
        for view in reversed(self.views[:-1]):
            flat_index = view.get_index(view.get_indices(flat_index))
        return flat_index

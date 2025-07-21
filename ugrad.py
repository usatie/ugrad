from __future__ import annotations
from typing import Self, Optional, Any
import numpy as np
from numpy.typing import NDArray


class Tensor:
    def __init__(
        self,
        data: NDArray[np.floating] | int | float,
        is_leaf: bool = True,
        requires_grad: bool = False,
        f: Optional["Function"] = None,
    ):
        self.data = (
            np.array(data, dtype=np.float64) if isinstance(data, (int, float)) else data
        )
        self.f = f
        self.grad: Optional[Tensor] = None
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    def __add__(self, other: Self | int | float) -> "Tensor":
        return Add.call(self, other)

    def __radd__(self, other: Self | int | float) -> "Tensor":
        return self + other

    def __mul__(self, other: Self | int | float) -> "Tensor":
        return Mul.call(self, other)

    def __rmul__(self, other: Self | int | float) -> "Tensor":
        return self * other

    def __neg__(self) -> "Tensor":
        return Neg.call(self)

    # self - other
    def __sub__(self, other: Self | int | float) -> "Tensor":
        return self + (-other)

    # other - self
    def __rsub__(self, other: Self | int | float) -> "Tensor":
        return (-self) + other

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def numpy(self) -> NDArray[np.floating]:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return self.data

    def matmul(self, other: Self) -> "Tensor":
        return Matmul.call(self, other)

    def sum(self) -> "Tensor":
        return Sum.call(self)

    def t(self) -> "Tensor":
        return Transpose.call(self)

    def backward(self, outgrad: Optional["Tensor"] = None) -> None:
        assert outgrad is not None or self.data.size == 1
        if self.f is None:
            return
        # Compute gradients
        if outgrad is None:
            grads = self.f.backward(Tensor(1.0))
        else:
            grads = self.f.backward(outgrad)
        grads = grads if isinstance(grads, tuple) else (grads,)
        # Single loop: initialize, update grads, and recurse
        for t, g in zip(self.f.inputs, grads):
            grad = Tensor(np.zeros_like(t.data))
            grad += g
            if t.requires_grad and t.is_leaf:
                # Update gradient
                if t.grad is None:
                    t.grad = grad
                else:
                    t.grad.data += grad.data
            # Recurse
            t.backward(t.grad if t.requires_grad and t.is_leaf else grad)


class Function:
    def forward(
        self, *args: "Tensor", **kwargs: Any
    ) -> NDArray[np.floating] | int | float:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(
        self, *args: "Tensor", **kwargs: Any
    ) -> "Tensor" | tuple["Tensor", ...]:
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __call__(
        self, *args: "Tensor", **kwargs: Any
    ) -> NDArray[np.floating] | int | float:
        self.inputs = args
        return self.forward(*args, **kwargs)

    def requires_grad(self) -> bool:
        return any([t.requires_grad for t in self.inputs])

    @classmethod
    def call(F, *args: Tensor | int | float) -> Tensor:
        f = F()  # Create fresh instance for each call
        tensors = (Tensor(x) if isinstance(x, (int, float)) else x for x in args)
        result = f(*tensors)
        return Tensor(result, f=f, is_leaf=False, requires_grad=f.requires_grad())


# mypy: disable-error-code="override"
class Add(Function):
    def forward(self, x: "Tensor", y: "Tensor") -> NDArray[np.floating] | int | float:
        return x.data + y.data

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        return out_grad, out_grad


class Neg(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating] | int | float:
        return -x.data

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return -out_grad


class Mul(Function):
    def forward(self, x: "Tensor", y: "Tensor") -> NDArray[np.floating] | int | float:
        return x.data * y.data

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        x, y = self.inputs
        x_grad = Tensor(out_grad.data * y.data)
        y_grad = Tensor(out_grad.data * x.data)
        return x_grad, y_grad


class Matmul(Function):
    def forward(self, x: "Tensor", y: "Tensor") -> NDArray[np.floating]:
        # x (2, 3)
        # y (3, 4)
        # out (2, 4)
        out = x.data.dot(y.data)
        return out

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        # x (2, 3)
        # y (3, 4)
        x, y = self.inputs
        # out_grad (2, 4)
        x_grad = Tensor(out_grad.data.dot(y.data.transpose()))
        y_grad = Tensor(x.data.transpose().dot(out_grad.data))
        return x_grad, y_grad


class Sum(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating] | int | float:
        return x.data.sum()

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return out_grad


class Transpose(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        return x.data.transpose()

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return Tensor(out_grad.data.T)


"""
class ReLU(Function):
    def forward(self):
        out = x.data
        out[out.data < 0] = 0
        return out
    
    def backward(self, out_grad):
        grad = out_grad
"""

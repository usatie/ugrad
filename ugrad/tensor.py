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
        return f"Tensor(data={self.data}, grad={self.grad})"

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

    # self ** n
    def __pow__(self, n: int) -> "Tensor":
        return Pow.call(self, n)

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

    @staticmethod
    def zeros(*shape) -> "Tensor":
        return Tensor(np.zeros(*shape))

    @staticmethod
    def zeros_like(a: "Tensor") -> "Tensor":
        return Tensor(np.zeros_like(a.data))

    def matmul(self, other: Self) -> "Tensor":
        return Matmul.call(self, other)

    def sum(self) -> "Tensor":
        return Sum.call(self)

    def t(self) -> "Tensor":
        return Transpose.call(self)

    def relu(self) -> "Tensor":
        return ReLU.call(self)

    def conv2d(self, filters: "Tensor") -> "Tensor":
        return Conv2D.call(self, filters)

    def backward(self, outgrad: Optional["Tensor"] = None) -> None:
        # print(f"[backward] self: {self.shape} ({self.data.dtype}), outgrad: {outgrad.shape if outgrad is not None else None} ({outgrad.data.dtype if outgrad is not None else None}), f: {self.f}")
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
            grad = Tensor(np.zeros_like(t.data, dtype=np.float64))
            if grad.data.shape != g.data.shape:
                g = unbroadcast(g, t.data.shape)
            grad += g
            if t.requires_grad and t.is_leaf:
                # Update gradient
                if t.grad is None:
                    t.grad = grad
                else:
                    t.grad.data += grad.data
            # Recurse
            t.backward(t.grad if t.requires_grad and t.is_leaf else grad)


# orig [1]            (, 1)
# x    [[1,1],[1,1]]  (2, 2)
def unbroadcast(x, shape):
    # Assume x is broadcasted from original shape
    x = x.data.copy()
    new_x = np.zeros_like(shape, dtype=np.float64)
    for i, dim in enumerate(reversed(x.shape)):
        orig_dim = shape[-i - 1] if i < len(shape) else 1
        if dim != orig_dim:
            x = x.sum(len(x.shape) - i - 1)
    return Tensor(x.reshape(shape))


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


class Pow(Function):
    def forward(self, x: "Tensor", n: int) -> NDArray[np.floating]:
        return x.data**n.data

    # x^n -> n * x^(n-1)
    def backward(self, out_grad: "Tensor") -> "Tensor":
        x, n = self.inputs
        x, n = x.data, n.data
        x_grad = Tensor((n * (x ** (n - 1))) * out_grad.data)
        return x_grad


class Sum(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating] | int | float:
        return x.data.sum()

    def backward(self, out_grad: "Tensor") -> "Tensor":
        assert out_grad.data.size == 1
        (x,) = self.inputs
        out = Tensor(np.ones_like(x.data, dtype=np.float64) * out_grad.data)
        return out


class Transpose(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        return x.data.transpose()

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return Tensor(out_grad.data.T)


class ReLU(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        out = x.data.copy()
        out[out < 0] = 0
        return out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x,) = self.inputs
        grad = out_grad.data.copy()
        grad[x.data < 0] = 0
        return Tensor(grad)


class Conv2D(Function):
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.kernels = Tensor(np.randn(out_channels, in_channels, kernel_size, kernel_size))
    """

    def forward(self, x: "Tensor", filters: "Tensor") -> "Tensor":
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        out_channels, in_channels, kernel_size, kernel_size = filters.shape
        N, cin, inH, inW = x.shape
        assert cin == in_channels
        pad = 0
        stride = 1
        outH = 1 + (inH - kernel_size + pad * 2) // stride
        outW = 1 + (inW - kernel_size + pad * 2) // stride
        out = np.zeros((N, out_channels, outH, outW))
        for j in range(outH):
            for i in range(outW):
                # dimension should be ok...
                # out[N, outc, y, x] = x[N, inc, y + ky, x + kx].dot(filters[outc, inc, ky, kx].t())
                for kj in range(kernel_size):
                    for ki in range(kernel_size):
                        out[:, :, j, i] += x.data[:, :, j + kj, i + ki].dot(
                            filters.data[:, :, kj, ki].T
                        )
        return out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        raise NotImplemented("Conv2D.backward")

from __future__ import annotations
from typing import Self, Optional, Any
import numpy as np
from numpy.typing import NDArray
from math import prod
import math

from ugrad.shape.shapetracker import ShapeTracker
from ugrad.shape.view import View


def calculate_gain(nonlinearity: str, a: float = 0.0) -> float:
    if nonlinearity == "relu":
        return np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return np.sqrt(2.0 / (1 + a**2))
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity is None or nonlinearity == "linear":
        return 1.0
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


class Tensor:
    def __init__(
        self,
        data: "Tensor" | NDArray[np.floating] | int | float | memoryview,
        st: ShapeTracker = None,
        is_leaf: bool = True,
        requires_grad: bool = False,
        f: Optional["Function"] = None,
    ):
        if isinstance(data, Tensor):
            self.dtype = data.dtype
            self.rawdata = data.rawdata
            self.st = data.st
        elif isinstance(data, memoryview):
            self.dtype = np.dtype(data.format)
            self.rawdata = data
            self.st = st if st is not None else ShapeTracker.create(data.shape)
        else:
            npdata = (
                np.array(data, dtype=np.float64)
                if isinstance(data, (int, float))
                else data
            )
            self.dtype = npdata.dtype
            mv = (
                npdata.data
                if npdata.flags.c_contiguous
                else np.ascontiguousarray(npdata).data
            )
            # Ensure the memoryview is 1D
            self.rawdata = mv.cast("B").cast(mv.format)
            self.st = st if st is not None else ShapeTracker.create(npdata.shape)
        self.f = f
        self.grad: Optional[Tensor] = None
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad

    @property
    def npdata(self) -> NDArray:
        return np.frombuffer(self.rawdata, dtype=self.dtype).reshape(self.st.shape)

    @property
    def ndim(self) -> int:
        return self.st.ndim

    def __getitem__(self, idx: int | tuple[int, ...]):
        if isinstance(idx, int):
            idx = (idx,)
        if any(not isinstance(i, int) for i in idx):
            raise TypeError("Only integer indexing is supported")
        if len(idx) > self.ndim:
            raise IndexError("Too many indices for tensor")
        if len(idx) == len(self.shape):
            # Maybe it's dumb to make flat index, if memoryview is multi dimensional, but we ensure data is 1D
            flat_index = self.st.get_index(idx)
            return self.rawdata[flat_index]
        else:
            return Tensor(self.rawdata, st=self.st.slice(idx))

    def tolist(self) -> list:
        def build_list(shape: tuple[int, ...], index_prefix: tuple[int, ...] = ()):
            if len(shape) == 0:
                return self.rawdata[0]
            elif len(shape) == 1:
                return [self[index_prefix + (i,)] for i in range(shape[0])]
            else:
                return [
                    build_list(shape[1:], index_prefix + (i,)) for i in range(shape[0])
                ]

        return build_list(self.shape)

    def __repr__(self) -> str:
        return f"Tensor(data={self.tolist()}, grad={self.grad})"

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
    def __pow__(self, n: int | float) -> "Tensor":
        return Pow.call(self, n)

    # self / other
    def __truediv__(self, other: "Tensor" | int | float) -> "Tensor":
        return self * (other**-1)

    # self < other
    def __lt__(self, other: "Tensor" | int | float) -> "Tensor":
        return Tensor(
            self.npdata < (other.npdata if isinstance(other, Tensor) else other)
        )

    # self <= other
    def __le__(self, other: "Tensor" | int | float) -> "Tensor":
        return Tensor(
            self.npdata <= (other.npdata if isinstance(other, Tensor) else other)
        )

    # self > other
    def __gt__(self, other: "Tensor" | int | float) -> "Tensor":
        return Tensor(
            self.npdata > (other.npdata if isinstance(other, Tensor) else other)
        )

    # self >= other
    def __ge__(self, other: "Tensor" | int | float) -> "Tensor":
        return Tensor(
            self.npdata >= (other.npdata if isinstance(other, Tensor) else other)
        )

    def assign(self, other: "Tensor" | int | float) -> "Tensor":
        if other.__class__ is not Tensor:
            other = Tensor(other)
        self.rawdata = other.rawdata
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.st.shape

    @property
    def size(self) -> int:
        return self.npdata.size

    def detach(self) -> "Tensor":
        return Tensor(self.npdata)

    def numpy(self) -> NDArray[np.floating]:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return self.npdata

    @staticmethod
    def zeros(*shape: int | tuple[int]) -> "Tensor":
        return Tensor(np.zeros(*shape))

    @staticmethod
    def zeros_like(a: "Tensor") -> "Tensor":
        return Tensor.zeros(a.shape)

    # Random samples from a uniform distribution over the interval [0, 1)
    @staticmethod
    def rand(*shape: int, **kwargs) -> "Tensor":
        return Tensor(np.random.rand(*shape), **kwargs)

    @staticmethod
    def randn(*shape: int, **kwargs) -> "Tensor":
        # Box-Muller transformation for generating standard normal distribution
        # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        u1 = 1 - Tensor.rand(*shape, **kwargs)  # (0, 1] is used to avoid log(0)
        u2 = Tensor.rand(*shape, **kwargs)
        R = (-2 * u1.log()).sqrt()
        theta = 2 * math.pi * u2
        z1 = R * theta.cos()
        return Tensor(z1, **kwargs)  # to ensure is_leaf=True

    @staticmethod
    def normal(*shape: int, mean: float = 0.0, std: float = 1.0, **kwargs) -> "Tensor":
        n = (std * Tensor.randn(*shape, **kwargs)) + mean
        return Tensor(n, **kwargs)  # to ensure is_leaf=True

    @staticmethod
    def uniform(*shape: int, low: float = 0.0, high: float = 1.0, **kwargs) -> "Tensor":
        u = ((high - low) * Tensor.rand(*shape, **kwargs)) + low
        return Tensor(u, **kwargs)  # to ensure is_leaf=True

    @staticmethod
    def xavier_uniform(*shape, gain=1.0, **kwargs) -> "Tensor":
        """
        Xavier/Glorot initialization
        Assuming shape is (out_ch, in_ch, ...).
        i.e. weight matrix is used in a transposed manner (x @ w.T)
        """

        fan_in = shape[1]
        fan_out = shape[0]
        if len(shape) > 2:
            # If more than 2 dimensions, use the product of the last dimensions
            receptive_field_size = prod(shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size
        a = gain * np.sqrt(6 / (fan_in + fan_out))  # Uniform distribution in [-a, a]
        return Tensor.uniform(*shape, low=-a, high=a, **kwargs)

    @staticmethod
    def xavier_normal(*shape, gain=1.0, **kwargs) -> "Tensor":
        """
        Xavier/Glorot initialization
        Assuming shape is (out_ch, in_ch, ...).
        i.e. weight matrix is used in a transposed manner (x @ w.T)
        """

        fan_in = shape[1]
        fan_out = shape[0]
        if len(shape) > 2:
            # If more than 2 dimensions, use the product of the last dimensions
            receptive_field_size = prod(shape[2:])
            fan_in *= receptive_field_size
            fan_out *= receptive_field_size
        std = gain * np.sqrt(2 / (fan_in + fan_out))
        return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

    @staticmethod
    def kaiming_uniform(*shape, a=0.0, nonlinearity="leaky_relu", **kwargs) -> "Tensor":
        """
        Kaiming/He initialization
        Assuming shape is (out_ch, in_ch, ...).
        i.e. weight matrix is used in a transposed manner (x @ w.T)
        """
        fan_in = shape[1]
        if len(shape) > 2:
            # If more than 2 dimensions, use the product of the last dimensions
            receptive_field_size = prod(shape[2:])
            fan_in *= receptive_field_size
        gain = calculate_gain(nonlinearity, a)
        a = gain * np.sqrt(3 / fan_in)
        return Tensor.uniform(*shape, low=-a, high=a, **kwargs)

    @staticmethod
    def kaiming_normal(*shape, a=0.0, nonlinearity="leaky_relu", **kwargs) -> "Tensor":
        """
        Kaiming/He initialization
        Assuming shape is (out_ch, in_ch, ...).
        i.e. weight matrix is used in a transposed manner (x @ w.T)
        """
        fan_in = shape[1]
        if len(shape) > 2:
            # If more than 2 dimensions, use the product of the last dimensions
            receptive_field_size = prod(shape[2:])
            fan_in *= receptive_field_size
        gain = calculate_gain(nonlinearity, a)
        std = gain / np.sqrt(fan_in)  # Normal distribution std
        return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

    def matmul(self, other: Self) -> "Tensor":
        return Matmul.call(self, other)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        return Sum.call(self, dim, keepdim)

    def t(self) -> "Tensor":
        return Transpose.call(self)

    def relu(self) -> "Tensor":
        return ReLU.call(self)

    def exp(self) -> "Tensor":
        return Exponential.call(self)

    def cos(self) -> "Tensor":
        return Cosine.call(self)

    def softmax(self, dim: int) -> "Tensor":
        e = self.exp()
        return e / e.sum(dim, keepdim=True)

    def log(self) -> "Tensor":
        return LogN.call(self)

    def unsqueeze(self, dim: int) -> "Tensor":
        return Unsqueeze.call(self, dim)

    def log_softmax(self, dim: int) -> "Tensor":
        return self - self.exp().sum(dim).log().unsqueeze(dim)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        out = self.sum(dim, keepdim=keepdim)
        N = prod(self.shape) / prod(out.shape)
        return out / N

    def square(self) -> "Tensor":
        return self**2

    def sqrt(self) -> "Tensor":
        return self**0.5

    def std(
        self, dim: Optional[int] = None, correction: int = 1, keepdim: bool = False
    ) -> "Tensor":
        meanzero = self - self.mean(dim, keepdim=True)
        sqsum = meanzero.square().sum(dim, keepdim=keepdim)
        N = prod(self.shape) / prod(sqsum.shape)
        out = sqsum / max(0, N - correction)
        return out.sqrt()

    def linear(self, weight: "Tensor", bias: Optional["Tensor"]) -> "Tensor":
        out = self.matmul(weight.t())
        return out + bias if bias else out

    def conv2d(self, filters: "Tensor") -> "Tensor":
        return Conv2D.call(self, filters)

    def backward(self, outgrad: Optional["Tensor"] = None) -> None:
        assert outgrad is not None or self.size == 1
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
            if not isinstance(t, Tensor):
                continue
            grad = Tensor.zeros_like(t)
            if grad.shape != g.shape:
                g = unbroadcast(g, grad.shape)
            grad += g
            if t.requires_grad and t.is_leaf:
                # Update gradient
                if t.grad is None:
                    t.grad = grad
                else:
                    t.grad.assign(t.grad + grad)
            # Recurse
            t.backward(t.grad if t.requires_grad and t.is_leaf else grad)


# orig [1]            (, 1)
# x    [[1,1],[1,1]]  (2, 2)
def unbroadcast(x: "Tensor", shape: tuple[int, ...]) -> "Tensor":
    # Assume x is broadcasted from original shape
    out = x.npdata.copy()
    i = 1
    while out.shape != shape and i <= len(out.shape):
        dim = out.shape[-i]
        orig_dim = shape[-i] if shape and i <= len(shape) else 1
        if dim != orig_dim:
            out = np.expand_dims(out.sum(-i), -i)
        else:
            i += 1
    return Tensor(out.reshape(shape))


class Function:
    def forward(self, *args: Any, **kwargs: Any) -> NDArray[np.floating] | int | float:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, out_grad: "Tensor") -> "Tensor" | tuple["Tensor", ...]:
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __call__(self, *args: Any, **kwargs: Any) -> NDArray[np.floating] | int | float:
        self.inputs = args
        return self.forward(*args, **kwargs)

    def requires_grad(self) -> bool:
        return any([isinstance(t, Tensor) and t.requires_grad for t in self.inputs])

    @classmethod
    def call(F, *args: Any) -> Tensor:
        f = F()  # Create fresh instance for each call
        result = f(*args)
        return Tensor(result, f=f, is_leaf=False, requires_grad=f.requires_grad())


def _broadcast_view(x, y):
    def pad(a, val):
        ndim = max(x.ndim, y.ndim)
        return (val,) * (ndim - len(a)) + a

    xshape = pad(x.shape, 1)
    yshape = pad(y.shape, 1)
    if any(
        xdim != 1 and ydim != 1 and xdim != ydim for xdim, ydim in zip(xshape, yshape)
    ):
        raise ValueError("shapes can't be broadcasted")
    shape = tuple(max(xdim, ydim) for xdim, ydim in zip(xshape, yshape))
    return View(shape, pad(x.strides, 0), x.offset), View(
        shape, pad(y.strides, 0), y.offset
    )


def broadcast_tensor(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    if x.shape == y.shape:
        return x, y
    vx, vy = _broadcast_view(x.st.views[-1], y.st.views[-1])
    return Tensor(x.rawdata, st=ShapeTracker(x.st.views[:-1] + (vx,))), Tensor(
        y.rawdata, st=ShapeTracker(y.st.views[:-1] + (vy,))
    )


def get_larger_format(fmt1: str, fmt2: str) -> str:
    if fmt1 == fmt2:
        return fmt1
    if fmt1 == "d" or fmt2 == "d":
        return "d"
    elif fmt1 == "f" or fmt2 == "f":
        return "f"
    elif fmt1 == "l" or fmt2 == "l":
        return "l"
    else:
        raise NotImplementedError(f"Unsupported format {fmt1} and {fmt2}")


def binary_op(x: Tensor, y: Tensor | int | float, op) -> Tensor:
    bx, by = broadcast_tensor(x, y if isinstance(y, Tensor) else Tensor(y))

    # If x and y have different format, convert to the bigger one (float > int)
    fmt = get_larger_format(bx.rawdata.format, by.rawdata.format)

    # Note: Currently, we always create a new array for the output, which may be very inefficient
    import array

    new_data = array.array(
        fmt,
        (
            op(bx[idx], by[idx])
            for idx in (bx.st.view.get_indices(i) for i in range(bx.size))
        ),
    )
    return Tensor(memoryview(new_data), st=bx.st)


def unary_op(x: Tensor, op) -> Tensor:
    # Note: Currently, we always create a new array for the output, which may be very inefficient
    import array

    new_data = array.array(
        x.rawdata.format, (op(x[x.st.view.get_indices(i)]) for i in range(x.size))
    )
    return Tensor(memoryview(new_data), st=x.st)


# mypy: disable-error-code="override"
class Add(Function):
    def forward(self, x: "Tensor", y: int | float | "Tensor") -> "Tensor":
        return binary_op(x, y, lambda a, b: a + b)

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        return out_grad, out_grad


class Neg(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating] | int | float:
        return unary_op(x, lambda a: -a)

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return -out_grad


class Mul(Function):
    def forward(self, x: "Tensor", y: int | float | "Tensor") -> NDArray[np.floating]:
        return binary_op(x, y, lambda a, b: a * b)

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        x, y = self.inputs
        other = y.npdata if isinstance(y, Tensor) else y
        x_grad = Tensor(out_grad.npdata * other)
        y_grad = Tensor(out_grad.npdata * x.npdata)
        return x_grad, y_grad


class Matmul(Function):
    def forward(self, x: "Tensor", y: "Tensor") -> NDArray[np.floating]:
        # x (2, 3)
        # y (3, 4)
        # out (2, 4)
        out = x.npdata.dot(y.npdata)
        return out

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        # x (2, 3)
        # y (3, 4)
        x, y = self.inputs
        # out_grad (2, 4)
        x_grad = Tensor(out_grad.npdata.dot(y.npdata.transpose()))
        y_grad = Tensor(x.npdata.transpose().dot(out_grad.npdata))
        return x_grad, y_grad


class Pow(Function):
    def forward(self, x: "Tensor", n: int | float) -> NDArray[np.floating]:
        if n == 0:
            return Tensor.ones_like(x)
        from math import nan

        return unary_op(x, lambda a: a ** n if abs(n) >= 1 or a >= 0 else nan)

    # x^n -> n * x^(n-1)
    def backward(self, out_grad: "Tensor") -> "Tensor":
        x, n = self.inputs
        x_grad = Tensor((n * (x.npdata ** (n - 1))) * out_grad.npdata)
        return x_grad


class Sum(Function):
    def forward(
        self, x: "Tensor", dim: Optional[int], keepdim: bool
    ) -> NDArray[np.floating] | int | float:
        if dim is None:
            out = x.npdata.sum(keepdims=keepdim)
            return out
        else:
            return x.npdata.sum(dim, keepdims=keepdim)

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x, dim, keepdim) = self.inputs
        out = Tensor(np.zeros_like(x.npdata))
        if dim is not None:
            if not keepdim:
                return out + out_grad.unsqueeze(dim)  # let numpy broadcast
            else:
                return out + out_grad
        else:
            return out + out_grad


class Unsqueeze(Function):
    def forward(self, x: "Tensor", dim: int) -> NDArray[np.floating]:
        return np.expand_dims(x.npdata, dim)

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x, dim) = self.inputs
        return Tensor(out_grad.npdata.squeeze(dim))


class Transpose(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        return x.npdata.transpose()

    def backward(self, out_grad: "Tensor") -> "Tensor":
        return Tensor(out_grad.npdata.T)


class ReLU(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        out = x.npdata.copy()
        out[out < 0] = 0
        return out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x,) = self.inputs
        grad = out_grad.npdata.copy()
        grad[x.npdata < 0] = 0
        return Tensor(grad)


class LogN(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        return np.log(x.npdata)

    def backward(self, out_grad: "Tensor") -> "Tensor":
        # y = log(x)
        # y' = 1 / x
        (x,) = self.inputs
        return Tensor(out_grad.npdata / x.npdata)


class Exponential(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        self.out = np.exp(x.npdata)
        return self.out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        # y = exp(x)
        # y' = exp(x)
        return Tensor(out_grad.npdata * self.out)


class Cosine(Function):
    def forward(self, x: "Tensor") -> NDArray[np.floating]:
        return np.cos(x.npdata)

    def backward(self, out_grad: "Tensor") -> "Tensor":
        # y = cos(x)
        # y' = -sin(x)
        (x,) = self.inputs
        return Tensor(-out_grad.npdata * np.sin(x.npdata))


class Conv2D(Function):
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.kernels = Tensor(np.randn(out_channels, in_channels, kernel_size, kernel_size))
    """

    def forward(self, x: "Tensor", filters: "Tensor") -> NDArray[np.floating]:
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
                # (N, outc, 1, 1) = (N, inc, ks, ks) ? (outc, inc, ks, ks)
                # (N, outc, 1, 1) = (N, inc * ks * ks) x (outc, inc * ks * ks).T
                left = x.npdata[:, :, j : j + kernel_size, i : i + kernel_size].reshape(
                    N, -1
                )
                right = filters.npdata.reshape(out_channels, -1)
                out[:, :, j, i] = left.dot(right.T)
        return out

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", ...]:
        x, filters = self.inputs
        out_channels, in_channels, kernel_size, kernel_size = filters.shape
        N, cin, inH, inW = x.shape
        x_grad = Tensor.zeros_like(x)
        filters_grad = Tensor.zeros_like(filters)
        pad = 0
        stride = 1
        outH = 1 + (inH - kernel_size + pad * 2) // stride
        outW = 1 + (inW - kernel_size + pad * 2) // stride
        # filters_grad[outc, inc, y, x]
        # outgrad[N, outc, y, x]
        # x[N, inc, y, x]
        for kj in range(kernel_size):
            for ki in range(kernel_size):
                # 1. filter (outc, inc, ks, ks)
                # filter(outc, inc, 1, 1) = out(N, outc, outH, outW) ? x(N, inc, outH, outW)
                # (outc, inc, 1, 1) = (outc, N * outH * outW) x (N * outH * outW, inc)
                left = out_grad.npdata.transpose(1, 0, 2, 3).reshape(out_channels, -1)
                right = x.npdata.transpose(0, 2, 3, 1)[
                    :, kj : outH + kj, ki : outW + ki, :
                ]
                right = right.reshape(-1, in_channels)
                filters_grad.npdata[:, :, kj, ki] += left.dot(right)
        # 2. x
        # (N, inc, inH, inW) = (N, outc, outH, outW) ? (outc, inc, ks, ks)
        # (N, inc, 1, 1) = (N, outc, ks, ks) ? (outc, inc, ks, ks)
        for j in range(outH):
            for i in range(outW):
                # dimension should be ok...
                for kj in range(kernel_size):
                    for ki in range(kernel_size):
                        # (N, inc, 1, 1) = (N, outc, 1, 1) x (outc, inc, 1, 1)
                        x_grad.npdata[:, :, j + kj, i + ki] += out_grad.npdata[
                            :, :, j, i
                        ].dot(filters.npdata[:, :, kj, ki])
        return x_grad, filters_grad

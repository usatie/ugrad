from __future__ import annotations

import array
import math
import random
import time
import os
from functools import reduce
from math import prod
from typing import Self, Optional, Any

import numpy as np
from numpy.typing import NDArray

from ugrad.shape.shapetracker import ShapeTracker
from ugrad.shape.view import View


dprint = lambda *args, **kwargs: (
    print(*args, **kwargs) if os.environ.get("DEBUG") else None
)
duration_realize = 0


def calculate_gain(nonlinearity: str, a: float = 0.0) -> float:
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + a**2))
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
        data: "Tensor" | NDArray | int | float | memoryview,
        st: ShapeTracker = None,
        is_leaf: bool = True,
        requires_grad: bool = False,
        f: Optional["Function"] = None,
        ops=tuple(),
    ):
        self.ops = ops
        if isinstance(data, Tensor):
            self._dtype = data.dtype
            self.rawdata = data.rawdata
            self.st = st if st is not None else data.st
            self.ops = ops if ops else data.ops
        elif isinstance(data, memoryview):
            self._dtype = np.dtype(data.format)
            self.rawdata = data
            self.st = st if st is not None else ShapeTracker.create(data.shape)
        elif isinstance(data, np.ndarray):
            self._dtype = data.dtype
            mv = (
                data.data
                if data.flags.c_contiguous
                else np.ascontiguousarray(data).data
            )
            # Ensure the memoryview is 1D
            self.rawdata = mv.cast("B").cast(mv.format)
            self.st = st if st is not None else ShapeTracker.create(data.shape)
        elif isinstance(data, (int, float)):
            self._dtype = np.float64
            self.rawdata = memoryview(array.array("d", [data]))
            self.st = st if st is not None else ShapeTracker.create(())
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        self.f = f
        self.grad: Optional[Tensor] = None
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad

    @property
    def dtype(self) -> np.dtype:
        return self.ops[-1].dtype if self.ops else self._dtype

    @property
    def fmt(self) -> str:
        fmt = self.ops[-1].fmt if self.ops else self.rawdata.format
        if fmt == "?":
            if self.dtype == np.bool:
                return "b"
            else:
                raise NotImplementedError("Unsupported bool format")
        return fmt

    def iterate_all_elements(self):
        for i in range(self.size):
            flat_index = self.st.get_flat_index(self.st.view.get_indices(i))
            yield self._getitem_by_flat_index(flat_index)

    def realize(self):
        global duration_realize
        dprint(
            f"Realizing Tensor with shape {self.shape} and ops {[op.name for op in self.ops]}"
        )
        if self.ops:
            start = time.time()
            new_data = array.array(self.fmt, self.iterate_all_elements())
            end = time.time()
            duration_realize += end - start
            self.rawdata = memoryview(new_data)
            self.ops = tuple()
            self.st = ShapeTracker.create(self.shape)
            dprint(
                f"Realize took {end - start:.6f}s, total time: {duration_realize:.6f}s"
            )

    @property
    def npdata(self) -> NDArray:
        if self.ops:
            # Note: Assume there are only element-wise ops
            # We can do lazy data loading here
            self.realize()
        np_strides = tuple(s * self.itemsize for s in self.st.view.strides)
        return np.lib.stride_tricks.as_strided(
            np.frombuffer(self.rawdata, dtype=self.dtype), self.shape, np_strides
        )

    @property
    def itemsize(self) -> int:
        return self.rawdata.itemsize

    @property
    def ndim(self) -> int:
        return self.st.ndim

    def reshape(self, *shape: int):
        return Reshape.call(self, *shape)

    def _getitem_by_flat_index(self, flat_index: int):
        return reduce(lambda x, op: op(flat_index), self.ops, self.rawdata[flat_index])

    def __getitem__(self, idx: int | tuple[int, ...]):
        if isinstance(idx, int):
            idx = (idx,)
        if any(not isinstance(i, int) for i in idx):
            raise TypeError("Only integer indexing is supported")
        if len(idx) > self.ndim:
            raise IndexError("Too many indices for tensor")
        if len(idx) == len(self.shape):
            # Maybe it's dumb to make flat index, if memoryview is multi dimensional, but we ensure data is 1D
            flat_index = self.st.get_flat_index(idx)
            return self._getitem_by_flat_index(flat_index)
        else:
            raise NotImplementedError("Subindexing is not supported yet")
            # return Tensor(self.rawdata, st=self.st.slice(idx))

    def tolist(self) -> list:
        def build_list(shape: tuple[int, ...], index_prefix: tuple[int, ...] = ()):
            if len(shape) == 0:
                return self[tuple()]
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
        dprint(
            f"Assigning Tensor with shape {self.shape} from other with shape {other.shape if isinstance(other, Tensor) else 'scalar'}"
        )
        if other.__class__ is not Tensor:
            other = Tensor(other)
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        assert self.fmt == other.fmt
        if other.ops:
            other.realize()
        self.rawdata = other.rawdata
        self.ops = tuple()
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.st.shape

    @property
    def size(self) -> int:
        return self.st.size

    def detach(self) -> "Tensor":
        return Tensor(self.npdata)

    def numpy(self) -> NDArray:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return self.npdata

    @staticmethod
    def empty(*shape: int | tuple[int]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        data = bytearray(prod(shape) * 8)
        out = Tensor(memoryview(data).cast("d"), st=ShapeTracker.create(shape))
        return out

    @staticmethod
    def zeros(*shape: int | tuple[int]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(
            memoryview(array.array("d", (0.0 for _ in range(prod(shape))))),
            st=ShapeTracker.create(shape),
        )

    @staticmethod
    def zeros_like(a: "Tensor") -> "Tensor":
        return Tensor.zeros(a.shape)

    # Random samples from a uniform distribution over the interval [0, 1)
    @staticmethod
    def rand(*shape: int, **kwargs) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(
            memoryview(array.array("d", (random.random() for _ in range(prod(shape))))),
            st=ShapeTracker.create(shape),
        )

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
        """
        # Orig implementation above using Tensor operations are 2x-3x slower than python inline code below
        # However, I think this approach is more important for future GPU implementation

        data = array.array("f")
        size = prod(shape)
        import random

        for _ in range(size):
            u1 = 1 - random.random()
            u2 = random.random()
            R = math.sqrt(-2 * math.log(u1))
            theta = 2 * math.pi * u2
            data.append(R * math.cos(theta))

        return Tensor(
            memoryview(data), st=ShapeTracker.create(shape), **kwargs
        )  # to ensure is_leaf=True
        """

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
        a = gain * math.sqrt(6 / (fan_in + fan_out))  # Uniform distribution in [-a, a]
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
        std = gain * math.sqrt(2 / (fan_in + fan_out))
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
        a = gain * math.sqrt(3 / fan_in)
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
        std = gain / math.sqrt(fan_in)  # Normal distribution std
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

    def sin(self) -> "Tensor":
        return Cosine.call(self - (math.pi / 2))

    def softmax(self, dim: int) -> "Tensor":
        e = self.exp()
        return e / e.sum(dim, keepdim=True)

    def log(self) -> "Tensor":
        return LogN.call(self)

    def squeeze(self, dim: int) -> "Tensor":
        return Squeeze.call(self, dim)

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
        # TODO: topological sort
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
    dprint(f"Unbroadcasting from {x.shape} to {shape}")
    # Assume x is broadcasted from original shape
    out = x
    i = 1
    while out.shape != shape and i <= len(out.shape):
        dim = out.shape[-i]
        orig_dim = shape[-i] if shape and i <= len(shape) else 1
        if dim != orig_dim:
            out = out.sum(-i).unsqueeze(-i)
        else:
            i += 1
    dprint(f"Unbroadcasted to {out.shape}")
    return Tensor(out, st=ShapeTracker.create(shape))


class Function:
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, out_grad: "Tensor") -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        self.inputs = args
        return self.forward(*args, **kwargs)

    def requires_grad(self) -> bool:
        return any([isinstance(t, Tensor) and t.requires_grad for t in self.inputs])

    @classmethod
    def call(F, *args: Any) -> Tensor:
        f = F()  # Create fresh instance for each call
        result = f(*args)
        requires_grad = f.requires_grad()
        out = Tensor(
            result, f=f, is_leaf=not requires_grad, requires_grad=requires_grad
        )
        return out


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
    return Tensor(
        x.rawdata, st=ShapeTracker(x.st.views[:-1] + (vx,)), ops=x.ops
    ), Tensor(y.rawdata, st=ShapeTracker(y.st.views[:-1] + (vy,)), ops=y.ops)


def get_larger_type(x: Tensor, y: Tensor) -> tuple[str, np.dtype]:
    if x.fmt == y.fmt:
        return x.fmt, x.dtype
    if x.fmt == "d" or y.fmt == "d":
        return "d", np.dtype("float64")
    elif fmt1 == "f" or fmt2 == "f":
        return "f", np.dtype("float32")
    elif fmt1 == "l" or fmt2 == "l":
        return "l", np.dtype("int64")
    else:
        raise NotImplementedError(f"Unsupported format {fmt1} and {fmt2}")


class Op:
    def __init__(self, fmt, dtype, f, name):
        self.fmt = fmt
        self.dtype = dtype
        self.f = f
        self.name = name

    def __call__(self, idx):
        return self.f(idx)


def binary_op(x: Tensor, y: Tensor | int | float, op, name=None) -> Tensor:
    bx, by = broadcast_tensor(x, y if isinstance(y, Tensor) else Tensor(y))

    # If x and y have different format, convert to the bigger one (float > int)
    fmt, dtype = get_larger_type(bx, by)
    out = Tensor(bx)

    # out.ops = (Op(fmt, dtype, lambda flat_index: print(f"{name}({(bxval := bx._getitem_by_flat_index(flat_index))}, {(byval := by[by.st.view.get_indices(flat_index)])})={(res := op(bxval, byval))} flat_index: {flat_index}, bx: {bx}, by: {by}") or res), )
    def f(flat_index):
        bxval = bx._getitem_by_flat_index(flat_index)
        byval = by[by.st.view.get_indices(flat_index)]
        res = op(bxval, byval)
        return res

    out.ops = (Op(fmt, dtype, f, name),)
    return out


def unary_op(x: Tensor, op, name=None) -> Tensor:
    # Note: Currently, we always create a new array for the output, which may be very inefficient
    out = Tensor(x)
    out.ops = (
        Op(
            x.fmt,
            x.dtype,
            lambda flat_index: op(x._getitem_by_flat_index(flat_index)),
            name,
        ),
    )
    return out


def reduce_op(x: Tensor, dim, keepdim, op, initial, name=None) -> Tensor:
    if dim is None:
        shape = tuple()
        if keepdim:
            shape = (1,) * x.ndim
        # Reduce all dimensions
        out = Tensor.empty(shape)
        cached_res = None

        def f(flat_index):
            nonlocal cached_res
            if cached_res is None:
                cached_res = reduce(op, x.iterate_all_elements(), initial)
            return cached_res

        out.ops = (Op(x.fmt, x.dtype, f, name),)
        return out
    else:
        if dim < 0:
            dim += x.ndim
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        out = Tensor.empty(tuple(shape))
        cached_res = {}

        def f(flat_index):
            nonlocal cached_res
            idx = out.st.view.get_indices(flat_index)
            if not keepdim:
                idx = idx[:dim] + (0,) + idx[dim:]
            if idx not in cached_res:
                # Compute reduction for this index, not using numpy features
                res = initial
                for i in range(x.shape[dim]):
                    full_idx = idx[:dim] + (i,) + idx[dim + 1 :]
                    val = x[full_idx]
                    res = op(res, val)
                cached_res[idx] = res
            return cached_res[idx]

        out.ops = (Op(x.fmt, x.dtype, f, name),)
        return out


# mypy: disable-error-code="override"
class Add(Function):
    def forward(self, x: "Tensor", y: int | float | Tensor) -> Tensor:
        dprint(
            f"Add.forward: x.shape={x.shape}, y.shape={y.shape if isinstance(y, Tensor) else 'scalar'}"
        )
        return binary_op(x, y, lambda a, b: a + b, "Add")

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        dprint("Add.backward {out_grad.shape=", out_grad.shape, "}")
        return out_grad, out_grad


class Neg(Function):
    def forward(self, x: "Tensor") -> Tensor:
        dprint("Neg.forward {x.shape=", x.shape, "}")
        return unary_op(x, lambda a: -a, "Neg")

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint("Neg.backward {out_grad.shape=", out_grad.shape, "}")
        return -out_grad


class Mul(Function):
    def forward(self, x: "Tensor", y: int | float | "Tensor") -> Tensor:
        dprint(
            f"Mul.forward: x.shape={x.shape}, y.shape={y.shape if isinstance(y, Tensor) else 'scalar'}"
        )
        return binary_op(x, y, lambda a, b: a * b, "Mul")

    def backward(self, out_grad: "Tensor") -> tuple["Tensor", "Tensor"]:
        x, y = self.inputs
        dprint(
            f"Mul.backward x.shape={x.shape}, y.shape={y.shape if isinstance(y, Tensor) else 'scalar'}, out_grad.shape={out_grad.shape}"
        )
        x_grad = Tensor(out_grad * y)
        y_grad = Tensor(out_grad * x)
        dprint(f"x_grad={x_grad}, y_grad={y_grad}")
        return x_grad, y_grad


class Matmul(Function):
    def forward(self, x: "Tensor", y: "Tensor") -> Tensor:
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
    def forward(self, x: "Tensor", n: int | float) -> Tensor:
        dprint(f"Pow.forward x.shape={x.shape}, n={n}")
        if n == 0:
            return Tensor.ones_like(x)

        return unary_op(
            x, lambda a: a ** n if abs(n) >= 1 or a >= 0 else math.nan, "Pow"
        )

    # x^n -> n * x^(n-1)
    def backward(self, out_grad: "Tensor") -> "Tensor":
        x, n = self.inputs
        dprint(
            f"Pow.backward x.shape={x.shape}, n={n}, out_grad.shape={out_grad.shape}"
        )
        x_grad = out_grad * n * (x.detach() ** (n - 1))
        return x_grad


class Sum(Function):
    def forward(self, x: "Tensor", dim: Optional[int], keepdim: bool) -> Tensor:
        dprint(f"Sum.forward: x.shape={x.shape}, dim={dim}, keepdim={keepdim}")
        return reduce_op(x, dim, keepdim, lambda a, b: a + b, 0, "Sum")

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x, dim, keepdim) = self.inputs
        dprint(
            f"Sum.backward: x.shape={x.shape}, dim={dim}, keepdim={keepdim}, out_grad.shape={out_grad.shape}"
        )
        out = Tensor.zeros_like(x)
        dprint(f"out = {out}, out_grad = {out_grad}")
        if dim is not None:
            if not keepdim:
                dprint(f"out_grad.unsqueeze(dim) = {out_grad.unsqueeze(dim)}")
                dprint(
                    f"out + out_grad.unsqueeze(dim) = {out + out_grad.unsqueeze(dim)}"
                )
                return out + out_grad.unsqueeze(dim)  # let numpy broadcast
            else:
                dprint(f"out + out_grad = {out + out_grad}")
                return out + out_grad
        else:
            dprint(f"out + out_grad = {out + out_grad}")
            return out + out_grad


class Reshape(Function):
    def forward(self, x: Tensor, *shape: tuple[int, ...]) -> Tensor:
        dprint(f"Reshape.forward: x.shape={x.shape}, shape={shape}")
        return Tensor(x, st=x.st.reshape(*shape))

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint(f"Reshape.backward: out_grad.shape={out_grad.shape}")
        (x, shape) = self.inputs
        return out_grad.reshape(*x.shape)


class Squeeze(Function):
    def forward(self, x: "Tensor", dim: int) -> Tensor:
        dprint(f"Squeeze.forward: x.shape={x.shape}, dim={dim}")
        # numpy.exceptions.AxisError: axis 2 is out of bounds for array of dimension 1
        # ValueError: cannot select an axis to squeeze out which has size not equal to one
        if dim >= x.ndim:
            raise ValueError(
                f"axis {dim} is out of bounds for tensor of dimension {x.ndim}"
            )
        if dim < 0:
            if dim + x.ndim < 0:
                raise ValueError(
                    f"axis {dim} is out of bounds for tensor of dimension {x.ndim}"
                )
            dim += x.ndim
        if x.shape[dim] != 1:
            raise ValueError("Cannot squeeze dimension with size not equal to one")
        return Tensor(x, st=x.st.reshape(x.shape[:dim] + x.shape[dim + 1 :]))

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint(f"Squeeze.backward: out_grad.shape={out_grad.shape}")
        (x, dim) = self.inputs
        return out_grad.unsqueeze(dim)


class Unsqueeze(Function):
    def forward(self, x: "Tensor", dim: int) -> Tensor:
        dprint(f"Unsqueeze.forward: x.shape={x.shape}, dim={dim}")
        # numpy.exceptions.AxisError: axis 2 is out of bounds for array of dimension 2
        ndim = x.ndim + 1
        if dim >= ndim:
            raise ValueError(
                f"axis {dim} is out of bounds for tensor of dimension {x.ndim}"
            )
        if dim < 0:
            if dim + ndim < 0:
                raise ValueError(
                    f"axis {dim} is out of bounds for tensor of dimension {x.ndim}"
                )
            dim += ndim
        return Tensor(x, st=x.st.reshape(x.shape[:dim] + (1,) + x.shape[dim:]))

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint(f"Unsqueeze.backward: out_grad.shape={out_grad.shape}")
        (x, dim) = self.inputs
        return out_grad.squeeze(dim)


class Transpose(Function):
    def forward(self, x: "Tensor") -> Tensor:
        dprint(f"Transpose.forward: x.shape={x.shape}")
        return Tensor(x, st=x.st.transpose())

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint(f"Transpose.backward: out_grad.shape={out_grad.shape}")
        return Tensor(out_grad, st=out_grad.st.transpose())


class ReLU(Function):
    def forward(self, x: "Tensor") -> Tensor:
        out = x.npdata.copy()
        out[out < 0] = 0
        return out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        (x,) = self.inputs
        grad = out_grad.npdata.copy()
        grad[x.npdata < 0] = 0
        return Tensor(grad)


class LogN(Function):
    def forward(self, x: "Tensor") -> Tensor:
        dprint(f"LogN.forward: x.shape={x.shape}")
        return unary_op(x, lambda a: math.log(a) if a > 0 else math.nan, "LogN")

    def backward(self, out_grad: "Tensor") -> "Tensor":
        # y = log(x)
        # y' = 1 / x
        (x,) = self.inputs
        return Tensor(out_grad.npdata / x.npdata)


class Exponential(Function):
    def forward(self, x: "Tensor") -> Tensor:
        dprint(f"Exponential.forward: x.shape={x.shape}")
        self.out = unary_op(x, lambda a: math.exp(a), "Exponential")
        return self.out

    def backward(self, out_grad: "Tensor") -> "Tensor":
        dprint(
            f"Exponential.backward: out_grad.shape={out_grad.shape}, self.out.shape={self.out.shape}"
        )
        # y = exp(x)
        # y' = exp(x)
        return out_grad * self.out


class Cosine(Function):
    def forward(self, x: "Tensor") -> Tensor:
        dprint(f"Cosine.forward: x.shape={x.shape}")
        return unary_op(x, lambda a: math.cos(a), "Cosine")

    def backward(self, out_grad: "Tensor") -> "Tensor":
        # y = cos(x)
        # y' = -sin(x)
        (x,) = self.inputs
        return out_grad * (-x.detach().sin())


class Conv2D(Function):
    def forward(self, x: "Tensor", filters: "Tensor") -> Tensor:
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
        npdata = filters_grad.npdata
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
                npdata[:, :, kj, ki] += left.dot(right)
        filters_grad = Tensor(npdata)
        # 2. x
        # (N, inc, inH, inW) = (N, outc, outH, outW) ? (outc, inc, ks, ks)
        # (N, inc, 1, 1) = (N, outc, ks, ks) ? (outc, inc, ks, ks)
        npdata = x_grad.npdata
        for j in range(outH):
            for i in range(outW):
                # dimension should be ok...
                for kj in range(kernel_size):
                    for ki in range(kernel_size):
                        # (N, inc, 1, 1) = (N, outc, 1, 1) x (outc, inc, 1, 1)
                        npdata[:, :, j + kj, i + ki] += out_grad.npdata[:, :, j, i].dot(
                            filters.npdata[:, :, kj, ki]
                        )
        x_grad = Tensor(npdata)
        return x_grad, filters_grad

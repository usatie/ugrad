from typing import Self

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other: Self|int|float):
        if (isinstance(other, int|float)):
            other = Value(other)
        return Value(self.data + other.data)

    def __radd__(self, other: Self|int|float):
        return self + other

    def __mul__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Value(other)
        return Value(self.data * other.data)

    def __rmul__(self, other: Self|int|float):
        return self * other

    # self - other
    def __sub__(self, other: Self):
        if (isinstance(other, int|float)):
            other = Value(other)
        return Value(self.data - other.data)

    # other - self
    def __rsub__(self, other: Self):
        return (-self) + other

    # -self
    def __neg__(self):
        return Value(-self.data)

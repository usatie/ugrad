from ugrad import Value

def test_add():
    a = Value(1)
    b = Value(2)
    c = a + b
    assert c.data == 3
    assert (a + 4).data == 5
    assert (4 + a).data == 5

def test_sub():
    a = Value(5)
    b = Value(3)
    c = a - b
    assert c.data == 2
    assert (a - 2).data == 3
    assert (2 - a).data == -3

def test_mul():
    a = Value(3)
    b = Value(4)
    c = a * b
    assert c.data == 12
    assert (a * 2).data == 6
    assert (2 * a).data == 6

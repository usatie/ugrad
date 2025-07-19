from ugrad import Value

if __name__ == '__main__':
    a = Value(1)
    b = Value(2)
    c = a + b
    assert c.data == 3
    d = c * c
    assert d.data == 9
    e = d - c
    assert e.data == 6

    assert (a + 2).data == 3
    assert (2 + a).data == 3
    assert (d - b).data == 7
    assert (d - 2).data == 7
    assert (2 - d).data == -7

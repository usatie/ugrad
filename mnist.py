import gzip, os
import numpy as np
from ugrad import Tensor
from ugrad.optim import SGD


def fetch_mnist():
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    BASE = os.path.dirname(__file__) + "/extra/datasets"
    X_train = (
        parse(BASE + "/mnist/train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse(BASE + "/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = (
        parse(BASE + "/mnist/t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse(BASE + "/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test


class Linear:
    def __init__(self, in_channel, out_channel, bias=True, activation="relu"):
        self.W = Tensor(
            np.random.normal(0, 1, (in_channel, out_channel)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channel), requires_grad=True) if bias else None
        self.activation = activation

    def __call__(self, x):
        # x : (bs, in_channel)
        x = x.matmul(self.W)
        if self.bias:
            x += self.bias
        if self.activation:
            return getattr(x, self.activation)()
        else:
            return x

    def parameters(self):
        if self.bias:
            return [self.W, self.bias]
        else:
            return [self.W]


class MLP:
    def __init__(self, channels, bias=True, activation="relu"):
        # channels : (784, 32, 16, 1)
        nl = len(channels) - 1
        self.layers = [
            Linear(*channels[i : i + 2], bias, activation if i < nl - 1 else None)
            for i in range(nl)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        print("before softmax")
        print(x.data[0])
        x = x.softmax(1)
        print("after softmax")
        print(x.data[0])
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


def main():
    model = MLP((784, 10))
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    print(Y_train.shape)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    def loss_fn(y_pred, y):
        bs = y.shape[0]
        classes = y_pred.shape[-1]
        idx = (range(bs), tuple([x.item() for x in y.data]))
        y_gt = Tensor(np.zeros((bs, classes)))
        y_gt.data[idx] = 1
        eps = 1e-6
        return (-((eps + y_pred * y_gt).log())).sum() * (1 / bs)

    n_epochs = 100000
    batch_size = 32
    idx = np.random.randint(0, len(X_train), batch_size)
    for i in range(n_epochs):
        X = X_train[idx]
        Y = Y_train[idx]

        # compute
        optimizer.zero_grad()
        out = model(Tensor(X))
        loss = loss_fn(out, Tensor(Y))
        if i % 1000 == 0:
            print(loss.data.item())
        loss.backward()
        print(model.layers[-1].bias)
        exit()
        optimizer.step()
    print(model(Tensor(X_train)).data.shape)


if __name__ == "__main__":
    main()

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
            x = l(x).batch_norm()
        x = x.log_softmax(-1)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

def transform_target(target, n_classes: int):
    bs = target.shape[0]
    idx = (range(bs), tuple([x.item() for x in target]))
    transformed = np.zeros((bs, n_classes))
    transformed[idx] = 1
    return transformed

def evaluate(model, X_test, Y_test):
    y_pred = model(Tensor(X_test))
    accuracy = (y_pred.data.argmax(-1) == Y_test).sum() / len(X_test)
    print(f"Accuracy: {accuracy}")


def main():
    model = MLP((784, 128, 32, 10))
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    def loss_fn(y_pred, y):
        bs = y.shape[0]
        n_classes = y_pred.shape[-1]
        y_gt = Tensor(transform_target(y.data, n_classes))
        eps = 1e-6
        return (y_gt - y_pred).sum() * (1 / bs)

    n_epochs = 100000
    batch_size = 32
    idx = np.random.randint(0, len(X_train), batch_size)
    print("-----Initial Evaluation-----")
    evaluate(model, X_test, Y_test)
    for i in range(n_epochs):
        X = X_train[idx]
        Y = Y_train[idx]

        # compute
        optimizer.zero_grad()
        out = model(Tensor(X))
        loss = loss_fn(out, Tensor(Y))
        if i % 100 == 0:
            print(loss.data.item())
            print("-----Evaluation-----")
            evaluate(model, X_test, Y_test)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()

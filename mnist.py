import gzip, os
import numpy as np
from ugrad import Tensor
from ugrad.optim import SGD
import tqdm


def fetch_mnist():
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    BASE = os.path.dirname(__file__) + "/extra/datasets"
    # Normalize the data to [0, 1]
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

def preprocess(X_train, Y_train, X_test, Y_test):
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return (
        X_train.astype(np.float32),
        Y_train.astype(np.int64),
        X_test.astype(np.float32),
        Y_test.astype(np.int64),
    )

class Linear:
    def __init__(self, in_channel, out_channel, bias=True, activation="relu"):
        # np.random.normal(0,1) was too large, causing gradient explosion
        # Xavier initialization
        self.W = Tensor(
            np.random.normal(0, np.sqrt(2.0/in_channel), (in_channel, out_channel)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channel), requires_grad=True) if bias else None
        self.activation = activation

    def __call__(self, x):
        # x : (bs, in_channel)
        x = x.matmul(self.W)
        if self.bias is not None:
            x += self.bias
        if self.activation:
            return getattr(x, self.activation)()
        else:
            return x

    def parameters(self):
        if self.bias is not None:
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
        for i, l in enumerate(self.layers):
            x = l(x)
        x = x.log_softmax(-1)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

def transform_target(target, n_classes: int):
    bs = target.shape[0]
    idx = (range(bs), tuple([int(x.item()) for x in target]))
    transformed = np.zeros((bs, n_classes))
    transformed[idx] = 1
    return transformed

def evaluate(model, X_test, Y_test):
    y_pred = model(Tensor(X_test))
    accuracy = (y_pred.data.argmax(-1) == Y_test).sum() / len(X_test)
    return accuracy


def main():
    model = MLP((784, 128, 32, 10))
    X_train, Y_train, X_test, Y_test = preprocess(*fetch_mnist())

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Negative log likelihood loss
    def loss_fn(y_pred, y):
        n_classes = y_pred.shape[-1]
        y_gt = Tensor(transform_target(y.data, n_classes))
        return -(y_gt * y_pred).mean()

    n_steps = 10000
    batch_size = 32
    for step in tqdm.tqdm(range(n_steps)):
        idx = np.random.randint(0, len(X_train), batch_size)
        X = X_train[idx]
        Y = Y_train[idx]

        # compute
        optimizer.zero_grad()
        out = model(Tensor(X))
        loss = loss_fn(out, Tensor(Y))
        if step % 1000 == 0:
            print(f"-----Evaluation (Step: {step})-----")
            print(f"loss: {loss.data.item():.4f}")
            accuracy = evaluate(model, X_test, Y_test)
            print(f"Accuracy: {accuracy}")
        loss.backward()

        optimizer.step()
    assert evaluate(model, X_test, Y_test) >= 0.95, "Model did not reach the expected accuracy."


if __name__ == "__main__":
    main()

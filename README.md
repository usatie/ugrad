# ugrad - micro(μ) grad framework (or usatie's grad framework)
This is a toy project for my learning, inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad).

I am trying to implement an autgrad engine with device backend support, something inbetween [micrograd (<150 lines)](https://github.com/karpathy/micrograd) and [teenygrad (<1000 lines)](https://github.com/tinygrad/teenygrad), but much less than [tinygrad (~10000 lines)](https://github.com/tinygrad/tinygrad) and [pytorch (~1,000,000 lines)](https://github.com/pytorch/pytorch)

## Usage
How to run unit tests.
```bash
pip install numpy torch pytest
python -m pytest
```

## Understanding autograd
Forward pass:
```
out.data = func.forward(inputs.data)
```

Backward pass:
```
inputs.grads = func.backward(out.grads)
```

So, in order to be able to compute the backward pass and store the grads, we need to store at least `func` and `in` inside the tensor. Which is called context in pytorch, and ctx in tinygrad. Sometimes more stuff like `out.data` and `exp(input)` are stored as well for convenience. In tinygrad, the Function class stores inputs and the instance of Function class is stored in tensor as `ctx`. 

This approach was not intuitive for me at the first glance, but now it perfectly makes sense.


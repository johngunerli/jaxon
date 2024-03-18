# NNX - A Neural Network Library using JAX

NNX is a Python library for building and training neural networks, implemented using the JAX library for high-performance machine learning research. Leveraging the power of JAX, JAX-NN provides a flexible and efficient framework for defining, training, and running neural networks. The library includes a range of predefined layers, activation functions, and utilities to facilitate the development of custom neural network architectures.

## Core Features

### Model Definition

- **Sequential Models**: Define neural network models in a sequential manner, stacking layers one after the other.
- **Custom Layers**: Extend the base `Module` class to create custom layers for specialized use cases.

### Layers

- **Linear (Fully Connected)**: Implements a fully connected layer.
- **Conv2D**: Applies a 2D convolution over an input signal composed of several input planes.
- **ReLU**: Applies the rectified linear unit function element-wise.
- **Sigmoid**: Applies the sigmoid activation function element-wise.

### Utilities

- **Parameter Initialization**: Automatically initializes network parameters the first time a model is called.
- **Model Representation**: Print a string representation of the model, including layer types and parameters.

### Example Usage

```python
import nnx
from jax import random

# Define a simple neural network
class SimpleNN(nnx.Sequential):
    def __init__(self):
        super().__init__(
            nnx.Linear(10, 5), 
            nnx.ReLU(), 
            nnx.Linear(5, 1), 
            nnx.Sigmoid()
        )

# Initialize the model
model = SimpleNN()

# Prepare dummy input
rng = random.PRNGKey(0)
dummy_input = random.normal(rng, (1, 10))

# Forward pass
output = model(dummy_input)
print(output)

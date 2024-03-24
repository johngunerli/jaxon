import jax
import jax.numpy as jnp
from jax import random, lax

def current_device():
    return jax.devices()[0]

def set_device(device_name: str):
    jax.config.update('jax_platform_name', device_name)
    print("Set device to", device_name)

def cuda_is_available():
    return jax.devices("gpu") != []

def device_count():
    return len(jax.devices())

class Module:
    def __init__(self):
        self._modules = {}
        self._params = None

    def add_module(self, name, module):
        self._modules[name] = module

    def modules(self):
        return self._modules

    def parameters(self):
        for name, module in self._modules.items():
            if module._params is not None:
                yield from module._params
            else:
                yield from module.parameters()

    def __repr__(self):
        module_str = ",\n  ".join(
            [f"({name}): {str(mod)}" for name, mod in self._modules.items()]
        )
        return f"{self.__class__.__name__}(\n  {module_str}\n)"


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(f"module{idx}", module)

    def init_params(self, rng, input_shape):
        self._params = []
        for name, module in self._modules.items():
            input_shape, module_params = module.init_params(rng, input_shape)
            self._params.append(module_params)
            rng, _ = random.split(rng)
        return input_shape, self._params

    def __call__(self, inputs, params=None):
        if params is None:
            params = self._params
        for module, module_params in zip(self._modules.values(), params):
            inputs = module(inputs, module_params)
        return inputs


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def init_params(self, rng, input_shape):
        output_shape = (input_shape[0], self.out_features)
        k1, k2 = random.split(rng)
        w = random.normal(k1, (self.in_features, self.out_features)) * 1e-2
        b = random.normal(k2, (self.out_features,)) * 1e-2
        self._params = (w, b)
        return output_shape, (w, b)

    def __call__(self, inputs, params=None):
        if params is None:
            params = self._params
        w, b = params
        return jnp.dot(inputs, w) + b

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias=True)"


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def init_params(self, rng, input_shape):
        return input_shape, None

    def __call__(self, inputs, params=None):
        return jnp.maximum(0, inputs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def init_params(self, rng, input_shape):
        return input_shape, None

    def __call__(self, inputs, params=None):
        return jax.nn.sigmoid(inputs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Conv2D(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding="SAME"):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding

    def init_params(self, rng, input_shape):
        in_channels = input_shape[1]  # Assuming NCHW format
        kernel_shape = (self.out_channels, in_channels) + self.kernel_size
        W = random.normal(rng, kernel_shape) * 0.01
        b = random.normal(rng, (self.out_channels,)) * 0.01
        self._params = (W, b)
        output_shape = (
            input_shape[0],
            self.out_channels,
            input_shape[2],
            input_shape[3],
        )
        return output_shape, self._params

    def __call__(self, inputs, params=None):
        if params is None:
            params = self._params
        W, b = params
        return lax.conv_general_dilated(
            inputs,
            W,
            (self.stride, self.stride),
            self.padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        ) + b.reshape((1, self.out_channels, 1, 1))


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def init_params(self, rng, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3]), None

    def __call__(self, inputs, params=None):
        return jnp.reshape(inputs, (inputs.shape[0], -1))

    def __repr__(self):
        return f"{self.__class__.__name__}()"

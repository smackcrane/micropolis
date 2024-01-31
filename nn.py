import numpy as np
import copy

rng = np.random.default_rng()

class Module:
    """Base class for nn layers"""

    def parameters(self):
        return []

    def set_parameters(self, new_model):
        my_shapes = [p.shape for p in self.parameters()]
        new_shapes = [p.shape for p in new_model.parameters()]
        assert my_shapes == new_shapes, f"Incompatible parameter shapes:\n{my_shapes}\n{new_shapes}"

        for my_p, new_p in zip(self.parameters(), new_model.parameters()):
            my_p[:] = new_p[:]

    def copy(self):
        return copy.deepcopy(self)

class Sequential(Module):
    """Sequential network"""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def add(self, layer):
        self.layers.append(layer)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        s = 'Sequential model\n'
        s += '\n'.join(['  ' + str(layer) for layer in self.layers])
        s += f'\ntotal parameters: {np.sum([np.prod(p.shape) for p in self.parameters()]):,}'
        return s

class Linear(Module):
    """Linear layer"""

    def __init__(self, in_dim, out_dim, bias=True):
        self.weight = rng.normal(size=(in_dim, out_dim)) / np.sqrt(in_dim)
        self.bias = rng.normal(size=(out_dim)) / np.sqrt(in_dim) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __repr__(self):
        s = f"Linear: weight {self.weight.shape}"
        if self.bias is not None:
            s += f", bias {self.bias.shape}"
        else:
            s += ", bias None"
        return s

class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, X):
        return np.where(X>0, X, 0)

    def __repr__(self):
        return "ReLU"

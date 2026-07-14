"""
Mini autograd - 0 dan yozilgan avtomatik differensiallash dvigateli.
PyTorch/TensorFlow YO'Q. Faqat numpy. Reverse-mode (backpropagation) o'zimizniki.

Tensor sinfi hisoblash grafini quradi va .backward() gradientlarni avtomatik hisoblaydi.
"""
import numpy as np


def _reduce_grad(grad, shape):
    """Broadcasting bo'lgan gradientni asl shaklga yig'adi."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _wrap(self, x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # --- qo'shish ---
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += _reduce_grad(out.grad, self.data.shape)
            other.grad += _reduce_grad(out.grad, other.data.shape)
        out._backward = _backward
        return out
    __radd__ = __add__

    # --- ko'paytirish (elementwise) ---
    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += _reduce_grad(out.grad * other.data, self.data.shape)
            other.grad += _reduce_grad(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))
    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    # --- matritsa ko'paytmasi (batched) ---
    def __matmul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _reduce_grad(ga, self.data.shape)
            other.grad += _reduce_grad(gb, other.data.shape)
        out._backward = _backward
        return out

    # --- bo'lish / daraja ---
    def __truediv__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data / other.data, (self, other), "/")
        def _backward():
            self.grad += _reduce_grad(out.grad / other.data, self.data.shape)
            other.grad += _reduce_grad(-out.grad * self.data / (other.data**2), other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += out.grad * p * (self.data ** (p - 1))
        out._backward = _backward
        return out

    def sqrt(self):
        return self ** 0.5

    # --- yig'indi / o'rtacha ---
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # --- aktivatsiyalar ---
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out

    def gelu(self):
        # tanh-approx GELU
        x = self.data
        c = np.sqrt(2/np.pi)
        inner = c * (x + 0.044715 * x**3)
        t = np.tanh(inner)
        val = 0.5 * x * (1 + t)
        out = Tensor(val, (self,), "gelu")
        def _backward():
            dinner = c * (1 + 3*0.044715*x**2)
            dt = (1 - t**2) * dinner
            grad = 0.5*(1+t) + 0.5*x*dt
            self.grad += out.grad * grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,), "softmax")
        def _backward():
            g = out.grad
            self.grad += p * (g - (g * p).sum(axis=axis, keepdims=True))
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, a, b):
        out = Tensor(np.swapaxes(self.data, a, b), (self,), "T")
        def _backward():
            self.grad += np.swapaxes(out.grad, a, b)
        out._backward = _backward
        return out

    # --- backprop ---
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def embed(table, idx):
    """Embedding qidirish: table[idx]. table=(V,E) Tensor, idx=(...,) int."""
    out = Tensor(table.data[idx], (table,), "embed")
    def _backward():
        np.add.at(table.grad, idx, out.grad)
    out._backward = _backward
    return out


def cross_entropy(logits, targets):
    """logits=(N,V) Tensor, targets=(N,) int. O'rtacha CE loss qaytaradi."""
    x = logits.data - logits.data.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    N = targets.shape[0]
    loss = -np.log(p[np.arange(N), targets] + 1e-9).mean()
    out = Tensor(loss, (logits,), "ce")
    def _backward():
        dp = p.copy()
        dp[np.arange(N), targets] -= 1
        dp /= N
        logits.grad += dp * out.grad
    out._backward = _backward
    return out

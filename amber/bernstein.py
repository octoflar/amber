#  Copyright (c) 2022. Ralf Quast
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow import Variable

tfk = tf.keras
tkc = tfk.constraints
tki = tfk.initializers
tkr = tfk.regularizers


class BPoly:
    """An n-variate Bernstein polynomial."""

    def __init__(self, d: ndarray):
        self._d, self._s = self._strides(d)

    def batch(self, c: ndarray, m: int) -> ndarray:
        assert np.shape(c) == (self._s[-1],)
        return np.repeat(c, m).reshape(c.shape + (m,))

    @tf.function(jit_compile=True)
    def __call__(self, b: ndarray, x: ndarray) -> ndarray:
        return self._op(self._d, self._s, b, x)

    @tf.function(jit_compile=True)
    def grad(self, b: np.ndarray, x: np.ndarray) -> tf.Tensor:
        return tf.gradients(self._op(self._d, self._s, b, x), x)[0]

    @staticmethod
    def _op(d: ndarray, s: ndarray, b: ndarray, x: ndarray) -> ndarray:
        n = d.size
        for i in range(n):
            for j in range(d[i] * s[i], 0, -s[i]):
                b = b[0:j] + (b[s[i]:j + s[i]] - b[0:j]) * x[i]
        return b[0]

    @staticmethod
    def _strides(d: ndarray) -> tuple[ndarray, ndarray]:
        n = d.size
        s = np.ones(n + 1, d.dtype)
        for i in reversed(range(n)):
            s[i - 1] = s[i] * (d[i] + 1)
        return d, s


class BLayer(tfk.layers.Layer):
    """An n-variate Bernstein layer."""
    _d: ndarray
    _s: ndarray
    _initializer: tki.Initializer
    _regularizer: tkr.Regularizer | None
    _trainable: bool
    _constraint: tkc.Constraint | None
    _c: Variable
    _m: int
    _n: int

    def __init__(self, d: ndarray,
                 initializer: tki.Initializer = tki.ones,
                 regularizer: tkr.Regularizer = None,
                 trainable: bool = True,
                 constraint: tkc.Constraint = None):
        super(BLayer, self).__init__()
        self._d, self._s = BLayer._strides(d)
        self._initializer = initializer
        self._regularizer = regularizer
        self._constraint = constraint
        self._trainable = trainable

    def build(self, input_shape):
        self._n, self._m = input_shape.as_list()
        self._c = self.add_weight(shape=self._s[-1],
                                  initializer=self._initializer,
                                  regularizer=self._regularizer,
                                  trainable=self._trainable,
                                  constraint=self._constraint)

    def call(self, inputs, **kwargs):
        b = BLayer._batch(self._c, self._m)
        return BLayer._op(self._d, self._s, b, inputs)

    def get_config(self) -> dict:
        return {"d": self._d,
                "initializer": self._initializer,
                "regularizer": self._regularizer,
                "trainable": self._trainable,
                "constraint": self._constraint}

    @staticmethod
    def _batch(c: Variable, m: int) -> Tensor:
        return tf.repeat(c, m).reshape(c.shape + (m,))

    @staticmethod
    def _op(d: ndarray, s: ndarray, b: Tensor, x: Variable) -> Tensor:
        n = d.size
        for i in range(n):
            for j in range(d[i] * s[i], 0, -s[i]):
                b = b[0:j] + (b[s[i]:j + s[i]] - b[0:j]) * x[i]
        return b[0]

    @staticmethod
    def _strides(d: ndarray) -> tuple[ndarray, ndarray]:
        n = d.size
        s = np.ones(n + 1, d.dtype)
        for i in reversed(range(n)):
            s[i - 1] = s[i] * (d[i] + 1)
        return d, s


class BInitializer(tki.Initializer):
    """To initialize a Bernstein layer with known coefficients."""
    _b: ndarray

    def __init__(self, d: ndarray, b: ndarray):
        assert np.shape(b) == np.product(d + 1)
        self._b = b

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.constant(self._b, dtype=dtype, shape=shape)

    def get_config(self):
        return {"b": self._b}

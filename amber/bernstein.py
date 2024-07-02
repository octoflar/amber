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
"""
A function and a layer to evaluate n-variate Bernstein polynomials
based on TensorFlow API.
"""

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

    _d: ndarray
    """The degrees of the n-variate Bernstein polynomial."""
    _m: ndarray
    """The strides within the n-variate Bernstein batch."""

    def __init__(self, d: ndarray):
        """
        Creates an n-variate Bernstein polynomial.

        :param d: The degrees of the n-variate Bernstein polynomial.
        """
        self._d = d
        self._m = _strides(d)

    def batch(self, c: ndarray, howmany: int) -> ndarray:
        """
        Creates an n-variate Bernstein batch.

        :param c: The Bernstein coefficients.
        :param howmany: The number of n-variate input vectors.
        :return: The Bernstein batch.
        """
        assert np.shape(c) == self._m[-1:]
        return np.repeat(c, howmany).reshape(c.shape + (howmany,))

    @tf.function(jit_compile=True)
    def __call__(self, b: ndarray, x: ndarray) -> ndarray | Tensor:
        """
        Evaluates the n-variate Bernstein polynomial.

        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial.
        """
        return _op(self._d, self._m, b, x)

    @tf.function(jit_compile=True)
    def grad(self, b: ndarray, x: ndarray) -> Tensor:
        """
        Evaluates the gradient of the n-variate Bernstein polynomial
        with respect to the input vectors.

        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the gradient of the Bernstein polynomial.
        """
        return tf.gradients(_op(self._d, self._m, b, x), x)[0]

    def eval(self, b: ndarray, x: ndarray) -> ndarray:
        """
        Evaluates the n-variate Bernstein polynomial.

        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial.
        """
        return _op(self._d, self._m, b, x)

    @property
    def strides(self) -> ndarray:
        """
        Returns the strides within the n-variate Bernstein batch.
        """
        return _strides(self._d)


class BLayer(tfk.layers.Layer):
    """An n-variate Bernstein layer."""

    _d: ndarray
    """The degrees of the Bernstein polynomial."""
    _m: ndarray
    """The strides within the n-variate Bernstein layer."""
    _initializer: tki.Initializer
    """The Bernstein coefficients initializer."""
    _regularizer: tkr.Regularizer | None
    """The Bernstein coefficients regularizer."""
    _trainable: bool
    """Are the Bernstein coefficients trainable?"""
    _constraint: tkc.Constraint | None
    """The constraints on the Bernstein coefficients."""
    _c: Variable
    """The Bernstein coefficients."""
    _howmany: int
    """The number of n-variate input vectors."""
    _n: int
    """The dimension of an input vector."""

    def __init__(
        self,
        d: ndarray,
        initializer: tki.Initializer = tki.ones,
        regularizer: tkr.Regularizer = None,
        trainable: bool = True,
        constraint: tkc.Constraint = None,
    ):
        """
        Creates a new Bernstein layer.

        :param d: The degrees of the Bernstein polynomial.
        :param initializer: An initializer.
        :param regularizer: A regularizer.
        :param trainable: Are the Bernstein coefficients trainable?
        :param constraint: A constraint.
        """
        super(BLayer, self).__init__()
        self._d = d
        self._m = _strides(d)
        self._initializer = initializer
        self._regularizer = regularizer
        self._constraint = constraint
        self._trainable = trainable

    def build(self, input_shape):
        """TensorFlow API."""
        self._n, self._howmany = input_shape
        self._c = self.add_weight(
            shape=self._m[-1:],
            initializer=self._initializer,
            regularizer=self._regularizer,
            trainable=self._trainable,
            constraint=self._constraint,
        )

    def call(self, inputs, **kwargs) -> Tensor:
        """TensorFlow API."""
        return _op(
            self._d, self._m, self._batch(self._c, self._howmany), inputs
        )

    def get_config(self) -> dict:
        """TensorFlow API."""
        return {
            "d": self._d,
            "initializer": self._initializer,
            "regularizer": self._regularizer,
            "trainable": self._trainable,
            "constraint": self._constraint,
        }

    @staticmethod
    def _batch(c: Variable, howmany: int) -> Tensor:
        """
        Returns a new Bernstein batch.

        :param c: The Bernstein coefficients.
        :param howmany: The number of n-variate input vectors.
        :return: The Bernstein batch.
        """
        return tf.repeat(c, howmany).reshape(c.shape + (howmany,))


def _op(
    d: ndarray, m: ndarray, b: ndarray | Tensor, x: ndarray | Variable
) -> ndarray | Tensor:
    """
    Performs the de Casteljau algorithm to evaluate an n-variate
    Bernstein polynomial.

    :param d: The degrees of the n-variate Bernstein polynomial.
    :param m: The strides within the Bernstein batch.
    :param b: The Bernstein batch.
    :param x: The n-variate input vectors.
    :return: The values of the Bernstein polynomial.
    """
    n = d.size
    for i in range(n):
        for j in reversed(range(m[i], m[i - 1], m[i])):
            b = b[0:j] + (b[m[i] : m[i] + j] - b[0:j]) * x[i]
    return b[0]


def _strides(d: ndarray) -> ndarray:
    """
    Computes the strides within an n-variate Bernstein batch.

    :param d: The degrees of the Bernstein polynomial.
    :return: The strides within the batch. The last element represents
    the size of the Bernstein batch.
    """
    n = d.size
    m = np.ones(n + 1, d.dtype)
    for i in reversed(range(n)):
        m[i - 1] = m[i] * (d[i] + 1)
    return m


class BInitializer(tki.Initializer):
    """
    Initializes an n-variate Bernstein layer.
    """

    _c: ndarray
    """The initial Bernstein coefficients."""

    def __init__(self, d: ndarray, c: ndarray):
        """
        Creates a new initializer.

        :param d: The degrees of the Bernstein polynomial.
        :param c: The Bernstein coefficients.
        """
        assert np.shape(c) == np.prod(d + 1)
        self._c = c

    def __call__(self, shape, dtype=None, **kwargs) -> Tensor:
        """TensorFlow API."""
        return tf.constant(self._c, dtype=dtype, shape=shape)

    def get_config(self):
        """TensorFlow API."""
        return {"c": self._c}

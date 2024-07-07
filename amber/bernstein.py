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
Evaluation of n-variate Bernstein polynomials based on TF API.
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
    """The strides within the n-variate batch of Bernstein coefficients."""

    def __init__(self, d: ndarray):
        """
        Creates an n-variate Bernstein polynomial.

        :param d: The degrees of the n-variate Bernstein polynomial.
        """
        self._d = d
        self._m = _strides(d)

    @tf.function(jit_compile=True)
    def __call__(
        self, c: ndarray | Variable, x: ndarray | Variable
    ) -> ndarray | Tensor:
        """
        Evaluates the n-variate Bernstein polynomial.

        The n-variate input is an array of shape ``(n, howmany)``. The batch
        of Bernstein coefficients is an array of either shape ``(m, howmany)``
        or ``(m, 1)`` where ``m = np.prod(d + 1)`` is the batch size defined
        by the degrees of the polynomial.

        :param c: The Bernstein coefficients.
        :param x: The n-variate input.
        :return: The values of the Bernstein polynomial.
        """
        return self.eval(c, x)

    @tf.function(jit_compile=True)
    def grad(self, c: ndarray | Variable, x: ndarray | Variable) -> Tensor:
        """
        Evaluates the gradient of the n-variate Bernstein polynomial
        with respect to the input vectors.

        :param c: The Bernstein coefficients.
        :param x: The n-variate input vectors.
        :return: The values of the gradient of the Bernstein polynomial.
        """
        return tf.gradients(self.eval(c, x), x)[0]

    def eval(
        self, c: ndarray | Variable, x: ndarray | Variable
    ) -> ndarray | Tensor:
        """
        Evaluates the n-variate Bernstein polynomial.

        :param c: The Bernstein coefficients.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial.
        """
        return _op(self._d, self._m, c, x)

    @property
    def strides(self) -> ndarray:
        """
        Returns the strides within the n-variate batch of Bernstein
        coefficients.

        :return: The strides. The last element represents the size
        of the batch of Bernstein coefficients.
        """
        return _strides(self._d)


class BLayer(tfk.layers.Layer):
    """An n-variate Bernstein layer."""

    _d: ndarray
    """The degrees of the Bernstein polynomial."""
    _m: ndarray
    """The strides within the n-variate batch of Bernstein coefficients."""
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
        """TF API."""
        self._n, self._howmany = input_shape
        self._c = self.add_weight(
            shape=(self._m[-1], 1),
            initializer=self._initializer,
            regularizer=self._regularizer,
            trainable=self._trainable,
            constraint=self._constraint,
        )

    def call(self, inputs, **kwargs) -> Tensor:
        """TF API."""
        return _op(self._d, self._m, self._c, inputs)

    def get_config(self) -> dict:
        """TF API."""
        return {
            "d": self._d,
            "initializer": self._initializer,
            "regularizer": self._regularizer,
            "trainable": self._trainable,
            "constraint": self._constraint,
        }


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
        """TF API."""
        return tf.constant(self._c, dtype=dtype, shape=shape)

    def get_config(self):
        """TF API."""
        return {"c": self._c}


@tf.function(jit_compile=True)
def _lerp(a, b, x):
    """
    Performs a linear interpolation.

    :param a: A value of the interpolant.
    :param b: A value of the interpolant.
    :param x: The interpolation weight.
    :return: The interpolated value.
    """
    return a + (b - a) * x


def _op(
    d: ndarray, m: ndarray, b: ndarray | Tensor, x: ndarray | Variable
) -> ndarray | Tensor:
    """
    Performs the de Casteljau algorithm to evaluate an n-variate
    Bernstein polynomial.

    :param d: The degrees of the n-variate Bernstein polynomial.
    :param m: The strides within the batch of Bernstein coefficients.
    :param b: The Bernstein coefficients.
    :param x: The n-variate input vectors.
    :return: The values of the Bernstein polynomial.
    """
    n = d.size
    for i in range(n):
        for j in reversed(range(m[i], m[i - 1], m[i])):
            b = _lerp(b[0:j], b[m[i] : m[i] + j], x[i])
    return b[0]


def _strides(d: ndarray) -> ndarray:
    """
    Computes the strides within an n-variate batch of Bernstein coefficients.

    :param d: The degrees of the Bernstein polynomial.
    :return: The strides. The last element represents the size of the batch.
    """
    n = d.size
    m = np.ones(n + 1, d.dtype)
    for i in reversed(range(n)):
        m[i - 1] = m[i] * (d[i] + 1)
    return m

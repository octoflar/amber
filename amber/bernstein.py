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

    _d: ndarray
    """The degrees of the n-variate Bernstein polynomial."""
    _s: ndarray
    """The strides within the n-variate Bernstein batch."""

    def __init__(self, d: ndarray):
        """Creates a new n-variate Bernstein polynomial of given degree.

        :param d: The degrees of the n-variate Bernstein polynomial.
        """
        self._d, self._s = self._strides(d)

    def batch(self, c: ndarray, m: int) -> ndarray:
        """Creates a new Bernstein batch for given Bernstein coefficients and
        a given number of n-variate input vectors.

        :param c: The Bernstein coefficients.
        :param m: The number of n-variate input vectors.
        :return: The corresponding Bernstein batch.
        """
        assert np.shape(c) == (self._s[-1],)
        return np.repeat(c, m).reshape(c.shape + (m,))

    @tf.function(jit_compile=True)
    def __call__(self, b: ndarray, x: ndarray) -> ndarray:
        """Evaluates the n-variate Bernstein polynomial for a given Bernstein
        batch and the given n-variate input vectors.

        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial for the given input
        vectors.
        """
        return self._op(self._d, self._s, b, x)

    @tf.function(jit_compile=True)
    def grad(self, b: ndarray, x: ndarray) -> Tensor:
        """Evaluates the gradient of the Bernstein polynomial for a given
        Bernstein batch and the given n-variate input vectors.

        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the gradient of the Bernstein polynomial for
        the given input vectors.
        """
        return tf.gradients(self._op(self._d, self._s, b, x), x)[0]

    @staticmethod
    def _op(d: ndarray, s: ndarray, b: ndarray, x: ndarray) -> ndarray:
        """Performs the de Casteljau algorithm to evaluate an n-variate
        Bernstein polynomial.

        :param d: The degrees of the n-variate Bernstein polynomial.
        :param s: The strides within the Bernstein batch.
        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial for the given input
        vectors.
        """
        n = d.size
        for i in range(n):
            for j in reversed(range(s[i], s[i - 1], s[i])):
                b = b[0:j] + (b[s[i] : s[i] + j] - b[0:j]) * x[i]
        return b[0]

    @staticmethod
    def _strides(d: ndarray) -> tuple[ndarray, ndarray]:
        """Computes the strides within the Bernstein coefficients array for
        an n-variate Bernstein polynomial of given degrees.

        :param d: The degrees of the n-variate Bernstein polynomial.
        :return: A tuple of the given degrees and the computed strides. The
        value of the last element of the strides array corresponds to the
        size of the coefficients array.
        """
        n = d.size
        s = np.ones(n + 1, d.dtype)
        for i in reversed(range(n)):
            s[i - 1] = s[i] * (d[i] + 1)
        return d, s


class BLayer(tfk.layers.Layer):
    """An n-variate Bernstein polynomial layer."""

    _d: ndarray
    """The degrees of the Bernstein polynomial."""
    _s: ndarray
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
    _m: int
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
        """Creates a new instance of this class.

        :param d: The degrees of the Bernstein polynomial.
        :param initializer: An initializer.
        :param regularizer: A regularizer.
        :param trainable: Are the Bernstein coefficients trainable?
        :param constraint: A constraint.
        """
        super(BLayer, self).__init__()
        self._d, self._s = BLayer._strides(d)
        self._initializer = initializer
        self._regularizer = regularizer
        self._constraint = constraint
        self._trainable = trainable

    def build(self, input_shape):
        self._n, self._m = input_shape
        self._c = self.add_weight(
            shape=self._s[-1],
            initializer=self._initializer,
            regularizer=self._regularizer,
            trainable=self._trainable,
            constraint=self._constraint,
        )

    def call(self, inputs, **kwargs) -> Tensor:
        return self._op(self._d, self._s, self._batch(self._c, self._m), inputs)

    def get_config(self) -> dict:
        return {
            "d": self._d,
            "initializer": self._initializer,
            "regularizer": self._regularizer,
            "trainable": self._trainable,
            "constraint": self._constraint,
        }

    @staticmethod
    def _batch(c: Variable, m: int) -> Tensor:
        """Returns a new Bernstein batch for given coefficients and number of
        n-variate input vectors.

        :param c: The Bernstein coefficients.
        :param m: The number of n-variate input vectors.
        :return: The Bernstein batch.
        """
        return tf.repeat(c, m).reshape(c.shape + (m,))

    @staticmethod
    def _op(d: ndarray, s: ndarray, b: Tensor, x: Variable) -> Tensor:
        """Performs the de Casteljau algorithm to evaluate an n-variate
        Bernstein polynomial.

        :param d: The degrees of the n-variate Bernstein polynomial.
        :param s: The strides within the Bernstein batch.
        :param b: The Bernstein batch.
        :param x: The n-variate input vectors.
        :return: The values of the Bernstein polynomial for the given input
        vectors.
        """
        n = d.size
        for i in range(n):
            for j in reversed(range(s[i], s[i - 1], s[i])):
                b = b[0:j] + (b[s[i] : s[i] + j] - b[0:j]) * x[i]
        return b[0]

    @staticmethod
    def _strides(d: ndarray) -> tuple[ndarray, ndarray]:
        """Computes the strides within the Bernstein coefficients array for
        an n-variate Bernstein polynomial of given degrees.

        :param d: The degrees of the n-variate Bernstein polynomial.

        :return: A tuple of the given degrees and the computed strides. The
        value of the last element of the strides array corresponds to the
        size of the coefficients array.
        """
        n = d.size
        s = np.ones(n + 1, d.dtype)
        for i in reversed(range(n)):
            s[i - 1] = s[i] * (d[i] + 1)
        return d, s


class BInitializer(tki.Initializer):
    """Initializes an n-variate Bernstein polynomial layer with initial
    coefficients."""

    _b: ndarray
    """The initial Bernstein coefficients."""

    def __init__(self, d: ndarray, b: ndarray):
        """Creates a new initializer to initialize an n-variate Bernstein
        polynomial layer of given degrees with given Bernstein coefficients.

        :param d: The degrees of the Bernstein polynomial layer.
        :param b: The Bernstein coefficients.
        """
        assert np.shape(b) == np.prod(d + 1)
        self._b = b

    def __call__(self, shape, dtype=None, **kwargs) -> Tensor:
        return tf.constant(self._b, dtype=dtype, shape=shape)

    def get_config(self):
        return {"b": self._b}

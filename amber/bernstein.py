#  Copyright (c) 2022. Ralf Quast
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np


class B:

    @staticmethod
    def basis(d: int, x: np.ndarray, y: np.ndarray):
        """Evaluates a univariate Bernstein basis at given coordinate
        values x in the unit interval.

        :param d: The degree of the Bernstein basis.
        :param x: The coordinate values.
        :param y: The evaluated Bernstein basis.
        """
        B.poly(d, np.identity(d + 1, dtype=x.dtype), x, y)

    @staticmethod
    def t1_basis(d: int, x: np.ndarray, t1_x: np.ndarray, y: np.ndarray, t1_y: np.ndarray):
        """The univariate tangent-linear Bernstein basis model.

        :param d: The degree of the Bernstein polynomial.
        :param x: The coordinate values.
        :param t1_x: The tangent-linear extension.
        :param y: The evaluated Bernstein basis.
        :param t1_y: The tangent-linear extension.
        """
        B.t1_poly(d, np.identity(d + 1, dtype=x.dtype), x, t1_x, y, t1_y)

    @staticmethod
    def poly(d: int, c: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Evaluates a univariate Bernstein polynomial at given
        coordinate values x in the unit interval.

        :param d: The degree of the Bernstein polynomial.
        :param c: The Bernstein coefficients.
        :param x: The coordinate values.
        :param y: The evaluated Bernstein polynomial.
        """
        m = np.shape(x)
        b = np.repeat(c, m).reshape(np.shape(c) + m)
        B._de_casteljau(d, b, x)
        y[:] = b[0]

    @staticmethod
    def t1_poly(d: int, c: np.ndarray,
                x: np.ndarray, t1_x: np.ndarray,
                y: np.ndarray, t1_y: np.ndarray):
        """The univariate tangent-linear Bernstein model.

        :param d: The degree of the Bernstein polynomial.
        :param c: The Bernstein coefficients.
        :param x: The coordinate values.
        :param t1_x: The tangent-linear extension.
        :param y: The evaluated Bernstein polynomial.
        :param t1_y: The tangent-linear extension.
        """
        m = np.shape(x)
        b = np.repeat(c, m).reshape(np.shape(c) + m)
        t1_b = np.zeros(np.shape(b))
        B._t1_de_casteljau(d, b, t1_b, x, t1_x)
        t1_y[:] = t1_b[0]
        y[:] = b[0]

    @staticmethod
    def poly_n(d: np.ndarray, c: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Evaluates an n-variate Bernstein polynomial at given coordinate
        vectors x in the unit n-cube.

        :param d: The degrees of the Bernstein polynomial.
        :param c: The Bernstein coefficients.
        :param x: The coordinate vectors.
        :param y: The evaluated Bernstein polynomial.
        """
        n, m = np.shape(x)
        b = np.repeat(c, m).reshape(np.shape(c) + (m,))
        B._de_casteljau_n(d, b, x)
        y[:] = b[0]

    @staticmethod
    def t1_poly_n(d: np.ndarray, c: np.ndarray,
                  x: np.ndarray, t1_x: np.ndarray,
                  y: np.ndarray, t1_y: np.ndarray):
        """The n-variate tangent-linear Bernstein model.

        :param d: The degrees of the Bernstein polynomial.
        :param c: The Bernstein coefficients.
        :param x: The coordinate values.
        :param t1_x: The tangent-linear extension.
        :param y: The evaluated Bernstein polynomial.
        :param t1_y: The tangent-linear extension.
        """
        n, m = np.shape(x)
        b = np.repeat(c, m).reshape(np.shape(c) + m)
        t1_b = np.zeros(np.shape(b))
        B._t1_de_casteljau_n(d, b, t1_b, x, t1_x)
        t1_y[:] = t1_b[0]
        y[:] = b[0]

    @staticmethod
    def _de_casteljau(d: int, b: np.ndarray, x: np.ndarray):
        """The univariate de Casteljau algorithm.

        :param d: The degree of the Bernstein polynomial.
        :param b: The Bernstein batch.
        :param x: The coordinate values.
        """
        for j in range(d, 0, -1):
            B._op(b, b, x, j, 1)

    @staticmethod
    def _t1_de_casteljau(d: int,
                         b: np.ndarray, t1_b: np.ndarray,
                         x: np.ndarray, t1_x: np.ndarray):
        """The univariate tangent-linear de Casteljau algorithm.

        :param d: The degree of the Bernstein polynomial.
        :param b: The Bernstein batch.
        :param t1_b: The tangent-linear extension.
        :param x: The coordinate values.
        :param t1_x: The tangent-linear extension.
        """
        for j in range(d, 0, -1):
            B._op(t1_b, t1_b, x, j, 1)
            B._op(t1_b, b, t1_x, j, 1)
            B._op(b, b, x, j, 1)

    @staticmethod
    def _de_casteljau_n(d: np.ndarray, b: np.ndarray, x: np.ndarray):
        """The n-variate de Casteljau algorithm.

        :param d: The degrees of the Bernstein polynomial.
        :param b: The Bernstein batch.
        :param x: The coordinate vectors.
        """
        s = B._strides(d)
        z = zip(d * s, s, x)
        for d, s, x, in z:
            for j in range(d, 0, -s):
                B._op(b, b, x, j, s)

    @staticmethod
    def _t1_de_casteljau_n(d: np.ndarray,
                           b: np.ndarray, t1_b: np.ndarray,
                           x: np.ndarray, t1_x: np.ndarray):
        """The n-variate tangent-linear de Casteljau algorithm.

        :param d: The degree of the Bernstein polynomial.
        :param b: The Bernstein batch.
        :param t1_b: The tangent-linear extension.
        :param x: The coordinate values.
        :param t1_x: The tangent-linear extension.
        """
        s = B._strides(d)
        z = zip(d * s, s, x, t1_x)
        for d, s, x, t1_x in z:
            for j in range(d, 0, -s):
                B._op(t1_b, t1_b, x, j, s)
                B._op(t1_b, b, t1_x, j, s)
                B._op(b, b, x, j, s)

    @staticmethod
    def _op(a: np.ndarray, b: np.ndarray, x: np.ndarray, j: int, s: int = 1):
        """The Bernstein operator.

        :param a: A Bernstein batch.
        :param b: A Bernstein batch.
        :param x: Coordinate values.
        :param j: An index.
        :param s: A stride.
        """
        a[0:j] += (b[s:j + s] - b[0:j]) * x

    @staticmethod
    def _strides(d: np.ndarray) -> np.ndarray:
        """Computes the strides in an n-variate Bernstein batch for an
        n-variate Bernstein polynomial of given degrees.

        :param d: The degrees of the Bernstein polynomial.
        :return: The strides within the batch.
        """
        n = d.size
        s = np.ones(n, dtype=d.dtype)
        for i in range(n - 1, 0, -1):
            s[i - 1] = s[i] * (d[i] + 1)
        return s

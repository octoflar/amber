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
            b[0:j] += B._dif(b, j) * x

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
            t1_b[0:j] += B._dif(t1_b, j) * x
            t1_b[0:j] += B._dif(b, j) * t1_x
            b[0:j] += B._dif(b, j) * x

    @staticmethod
    def _de_casteljau_n(d: np.ndarray, b: np.ndarray, x: np.ndarray):
        """The n-variate de Casteljau algorithm.

        :param d: The degrees of the Bernstein polynomial.
        :param b: The Bernstein batch.
        :param x: The coordinate vectors.
        """
        s = B._strides(d)
        for d_, s_, x_, in zip(d * s, s, x):
            for j in range(d_, 0, -s_):
                b[0:j] += B._dif(b, j, s_) * x_

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
        for d_, s_, x_, t1_x_ in zip(d * s, s, x, t1_x):
            for j in range(d_, 0, -s_):
                t1_b[0:j] += B._dif(t1_b, j, s_) * x_
                t1_b[0:j] += B._dif(b, j, s_) * t1_x_
                b[0:j] += B._dif(b, j, s_) * x_

    @staticmethod
    def _dif(b: np.ndarray, j: int, s: int = 1) -> np.ndarray:
        """The Bernstein difference operator.

        :param b: A Bernstein batch.
        :param j: An index.
        :param s: A stride.
        :return: A difference.
        """
        return b[s:j + s] - b[0:j]

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

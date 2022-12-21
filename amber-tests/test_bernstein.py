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

import unittest

import numpy as np
import tensorflow as tf

from bernstein import B

tfk = tf.keras


class BernsteinTest(unittest.TestCase):

    def setUp(self) -> None:
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()

    def test_basis(self):
        d = 4
        x = np.array([0.0, 0.5, 1.0])
        y = np.zeros((d + 1,) + np.shape(x))
        B.basis(d, x, y)

        self.assertAlmostEqual(1.0000, y[0, 0])
        self.assertAlmostEqual(0.0625, y[0, 1])
        self.assertAlmostEqual(0.0000, y[0, 2])
        self.assertAlmostEqual(0.0000, y[1, 0])
        self.assertAlmostEqual(0.2500, y[1, 1])
        self.assertAlmostEqual(0.0000, y[1, 2])
        self.assertAlmostEqual(0.0000, y[2, 0])
        self.assertAlmostEqual(0.3750, y[2, 1])
        self.assertAlmostEqual(0.0000, y[2, 2])
        self.assertAlmostEqual(0.0000, y[3, 0])
        self.assertAlmostEqual(0.2500, y[3, 1])
        self.assertAlmostEqual(0.0000, y[3, 2])
        self.assertAlmostEqual(0.0000, y[4, 0])
        self.assertAlmostEqual(0.0625, y[4, 1])
        self.assertAlmostEqual(1.0000, y[4, 2])

    def test_poly_zero_coefficients(self):
        d = 4
        c = np.zeros(d + 1)
        x = np.array([0.0, 0.5, 1.0])
        y = np.zeros(np.shape(x))

        B.poly(d, c, x, y)
        self.assertAlmostEqual(0.0, y[0], 6)
        self.assertAlmostEqual(0.0, y[1], 6)
        self.assertAlmostEqual(0.0, y[2], 6)

    def test_poly_unit_coefficients(self):
        d = 4
        c = np.ones(d + 1)
        x = np.array([0.0, 0.5, 1.0])
        y = np.zeros(np.shape(x))

        B.poly(d, c, x, y)
        self.assertAlmostEqual(1.0, y[0], 6)
        self.assertAlmostEqual(1.0, y[1], 6)
        self.assertAlmostEqual(1.0, y[2], 6)

    def test_poly_integral_coefficients(self):
        d = 4
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0.3141, 0.2718, 0.5772])
        y = np.zeros(np.shape(x))

        B.poly(d, c, x, y)
        self.assertAlmostEqual(2.2564, y[0], 6)
        self.assertAlmostEqual(2.0872, y[1], 6)
        self.assertAlmostEqual(3.3088, y[2], 6)

    def test_poly_1_zero_coefficients(self):
        n = 1
        m = 3
        d = np.full(n, 4)
        h = np.product(d + 1)
        c = np.zeros(h)
        x = np.array([0.0, 0.5, 1.0]).reshape(n, m)
        y = np.zeros(m)

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(0.0, y[0], 6)
        self.assertAlmostEqual(0.0, y[1], 6)
        self.assertAlmostEqual(0.0, y[2], 6)

    def test_poly_1_unit_coefficients(self):
        n = 1
        m = 3
        d = np.full(n, 4)
        h = np.product(d + 1)
        c = np.ones(h)
        x = np.array([0.0, 0.5, 1.0]).reshape(n, m)
        y = np.zeros(m)

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(1.0, y[0], 6)
        self.assertAlmostEqual(1.0, y[1], 6)
        self.assertAlmostEqual(1.0, y[2], 6)

    def test_poly_1_integral_coefficients(self):
        n = 1
        m = 3
        d = np.full(n, 4)
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0.3141, 0.2718, 0.5772]).reshape(n, m)
        y = np.zeros(m)

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(2.2564, y[0], 6)
        self.assertAlmostEqual(2.0872, y[1], 6)
        self.assertAlmostEqual(3.3088, y[2], 6)

    def test_poly_2_zero_coefficients(self):
        n = 2
        m = 5
        d = np.array([4, 3])
        h = np.product(d + 1)
        c = np.zeros(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        x[0, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        x[1, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.5])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(0.0, y[0], 6)
        self.assertAlmostEqual(0.0, y[1], 6)
        self.assertAlmostEqual(0.0, y[2], 6)
        self.assertAlmostEqual(0.0, y[3], 6)
        self.assertAlmostEqual(0.0, y[4], 6)

    def test_poly_2_unit_coefficients(self):
        n = 2
        m = 5
        d = np.array([4, 3])
        h = np.product(d + 1)
        c = np.ones(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        x[0, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        x[1, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.5])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(1.0, y[0], 6)
        self.assertAlmostEqual(1.0, y[1], 6)
        self.assertAlmostEqual(1.0, y[2], 6)
        self.assertAlmostEqual(1.0, y[3], 6)
        self.assertAlmostEqual(1.0, y[4], 6)

    def test_poly_2_integral_coefficients(self):
        n = 2
        m = 2
        d = np.array([4, 3])
        h = np.product(d + 1)
        c = np.zeros(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        BernsteinTest.fill(c)
        x[0, :] = np.array([0.2718, 0.5772])
        x[1, :] = np.array([0.5772, 0.2718])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(7.0804, y[0], 6)
        self.assertAlmostEqual(11.0506, y[1], 6)

    def test_poly_3_zero_coefficients(self):
        n = 3
        m = 9
        d = np.array([4, 3, 2])
        h = np.product(d + 1)
        c = np.zeros(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        x[0, :] = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5])
        x[1, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5])
        x[2, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.5])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(0.0, y[0], 6)
        self.assertAlmostEqual(0.0, y[1], 6)
        self.assertAlmostEqual(0.0, y[2], 6)
        self.assertAlmostEqual(0.0, y[3], 6)
        self.assertAlmostEqual(0.0, y[4], 6)
        self.assertAlmostEqual(0.0, y[5], 6)
        self.assertAlmostEqual(0.0, y[6], 6)
        self.assertAlmostEqual(0.0, y[7], 6)
        self.assertAlmostEqual(0.0, y[8], 6)

    def test_poly_3_unit_coefficients(self):
        n = 3
        m = 9
        d = np.array([4, 3, 2])
        h = np.product(d + 1)
        c = np.ones(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        x[0, :] = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5])
        x[1, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5])
        x[2, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.5])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(1.0, y[0], 6)
        self.assertAlmostEqual(1.0, y[1], 6)
        self.assertAlmostEqual(1.0, y[2], 6)
        self.assertAlmostEqual(1.0, y[3], 6)
        self.assertAlmostEqual(1.0, y[4], 6)
        self.assertAlmostEqual(1.0, y[5], 6)
        self.assertAlmostEqual(1.0, y[6], 6)
        self.assertAlmostEqual(1.0, y[7], 6)
        self.assertAlmostEqual(1.0, y[8], 6)

    def test_poly_3_integral_coefficients(self):
        n = 3
        m = 3
        d = np.array([4, 3, 2])
        h = np.product(d + 1)
        c = np.zeros(h)
        x = np.zeros((n, m))
        y = np.zeros(m)

        BernsteinTest.fill(c)
        x[0, :] = np.array([0.2718, 0.5772, 0.3141])
        x[1, :] = np.array([0.5772, 0.3141, 0.2718])
        x[2, :] = np.array([0.3141, 0.2718, 0.5772])

        B.poly_n(d, c, x, y)
        self.assertAlmostEqual(19.8694, y[0], 6)
        self.assertAlmostEqual(32.0761, y[1], 6)
        self.assertAlmostEqual(19.6774, y[2], 6)

    def test_t1_basis(self):
        d = 2
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        t1_x = np.ones(np.shape(x))
        y = np.zeros((d + 1,) + np.shape(x))
        t1_y = np.zeros(np.shape(y))

        B.t1_basis(d, x, t1_x, y, t1_y)
        self.assertAlmostEqual(-2.0, t1_y[0, 0])
        self.assertAlmostEqual(-1.5, t1_y[0, 1])
        self.assertAlmostEqual(-1.0, t1_y[0, 2])
        self.assertAlmostEqual(-0.5, t1_y[0, 3])
        self.assertAlmostEqual(0.0, t1_y[0, 4])
        self.assertAlmostEqual(2.0, t1_y[1, 0])
        self.assertAlmostEqual(1.0, t1_y[1, 1])
        self.assertAlmostEqual(0.0, t1_y[1, 2])
        self.assertAlmostEqual(-1.0, t1_y[1, 3])
        self.assertAlmostEqual(-2.0, t1_y[1, 4])
        self.assertAlmostEqual(0.0, t1_y[2, 0])
        self.assertAlmostEqual(0.5, t1_y[2, 1])
        self.assertAlmostEqual(1.0, t1_y[2, 2])
        self.assertAlmostEqual(1.5, t1_y[2, 3])
        self.assertAlmostEqual(2.0, t1_y[2, 4])

    def test_t1_poly(self):
        d = 2
        c = np.array([1.0, 1.0, 0.0])
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        t1_x = np.ones(np.shape(x))
        y = np.zeros(np.shape(x))
        t1_y = np.zeros(np.shape(x))
        B.t1_poly(d, c, x, t1_x, y, t1_y)
        self.assertAlmostEqual(0.0, t1_y[0])
        self.assertAlmostEqual(-0.5, t1_y[1])
        self.assertAlmostEqual(-1.0, t1_y[2])
        self.assertAlmostEqual(-1.5, t1_y[3])
        self.assertAlmostEqual(-2.0, t1_y[4])

    def test_layer(self):
        d = 4
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = B.Layer(d, c)

        x = np.array([0.3141, 0.2718, 0.5772])
        x = tf.Variable(x)
        y = f(x)
        self.assertAlmostEqual(2.2564, y[0], 6)
        self.assertAlmostEqual(2.0872, y[1], 6)
        self.assertAlmostEqual(3.3088, y[2], 6)

    def test_layer_gradient(self):
        d = 2
        c = np.array([1.0, 1.0, 0.0])
        f = B.Layer(d, c)

        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x = tf.Variable(x)
        with tf.GradientTape() as t:
            y = f(x)
        g = t.gradient(y, x)
        self.assertAlmostEqual(0.0, g[0])
        self.assertAlmostEqual(-0.5, g[1])
        self.assertAlmostEqual(-1.0, g[2])
        self.assertAlmostEqual(-1.5, g[3])
        self.assertAlmostEqual(-2.0, g[4])

    def test_tf_function(self):
        d = 4
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0.3141, 0.2718, 0.5772])
        b = B.Poly.batch(c, np.shape(x))

        f = B.Poly()
        y = f(d, b, x)
        self.assertAlmostEqual(2.2564, y[0])
        self.assertAlmostEqual(2.0872, y[1])
        self.assertAlmostEqual(3.3088, y[2])

    def test_tf_function_gradient(self):
        d = 2
        c = np.array([1.0, 1.0, 0.0])
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        b = B.Poly.batch(c, np.shape(x))

        f = B.Poly()
        g = f.grad(d, b, x)
        self.assertAlmostEqual(0.0, g[0])
        self.assertAlmostEqual(-0.5, g[1])
        self.assertAlmostEqual(-1.0, g[2])
        self.assertAlmostEqual(-1.5, g[3])
        self.assertAlmostEqual(-2.0, g[4])

    @staticmethod
    def fill(b: np.ndarray):
        """Consecutively fills an n-variate Bernstein batch with integral
        numbers 1, 2, 3, ...

        :param b: The Bernstein batch.
        """
        for i in range(b.size):
            b[i] = i + 1
        return


if __name__ == '__main__':
    unittest.main()

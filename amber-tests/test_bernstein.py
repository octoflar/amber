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

import unittest

import numpy as np
import tensorflow as tf

from bernstein import BInitializer
from bernstein import BLayer
from bernstein import BPoly

tfk = tf.keras


class BPolyTest(unittest.TestCase):

    def setUp(self) -> None:
        tf.experimental.numpy.experimental_enable_numpy_behavior()

    def test_b_poly(self):
        d = np.array([4, 3, 2])
        c = np.arange(np.prod(d + 1)) + 1.0
        x = np.array(
            [
                [0.2718, 0.5772, 0.3141],
                [0.5772, 0.3141, 0.2718],
                [0.3141, 0.2718, 0.5772],
            ]
        )
        n, m = np.shape(x)
        f = BPoly(d)
        b = f.batch(c, m)

        y = f.eval(b, x).data
        self.assertAlmostEqual(19.8694, y[0])
        self.assertAlmostEqual(32.0761, y[1])
        self.assertAlmostEqual(19.6774, y[2])

        y = f(b, x)
        self.assertAlmostEqual(19.8694, y[0])
        self.assertAlmostEqual(32.0761, y[1])
        self.assertAlmostEqual(19.6774, y[2])

    def test_b_poly_gradient(self):
        d = np.array([2, 2])
        c = np.arange(np.prod(d + 1)) + 1.0
        x = np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]])
        n, m = np.shape(x)
        f = BPoly(d)
        b = f.batch(c, m)

        y = f(b, x)
        self.assertAlmostEqual(1.0, y[0])
        self.assertAlmostEqual(3.0, y[1])
        self.assertAlmostEqual(5.0, y[2])
        self.assertAlmostEqual(7.0, y[3])
        self.assertAlmostEqual(9.0, y[4])

        g = f.grad(b, x).data
        self.assertAlmostEqual(6.0, g[0, 0])
        self.assertAlmostEqual(6.0, g[0, 1])
        self.assertAlmostEqual(6.0, g[0, 2])
        self.assertAlmostEqual(6.0, g[0, 3])
        self.assertAlmostEqual(6.0, g[0, 4])
        self.assertAlmostEqual(2.0, g[1, 0])
        self.assertAlmostEqual(2.0, g[1, 1])
        self.assertAlmostEqual(2.0, g[1, 2])
        self.assertAlmostEqual(2.0, g[1, 3])
        self.assertAlmostEqual(2.0, g[1, 4])


class BLayerTest(unittest.TestCase):

    def setUp(self) -> None:
        tf.experimental.numpy.experimental_enable_numpy_behavior()

    def test_b_layer(self):
        d = np.array([4, 3, 2])
        c = np.arange(np.prod(d + 1)) + 1.0
        x = tf.Variable(
            [
                [0.2718, 0.5772, 0.3141],
                [0.5772, 0.3141, 0.2718],
                [0.3141, 0.2718, 0.5772],
            ]
        )
        f = BLayer(d, BInitializer(d, c))

        y = f(x).data
        self.assertAlmostEqual(19.8694, y[0], 4)
        self.assertAlmostEqual(32.0761, y[1], 4)
        self.assertAlmostEqual(19.6774, y[2], 4)

    def test_b_layer_gradient(self):
        d = np.array([2, 2])
        c = np.arange(np.prod(d + 1)) + 1.0
        x = tf.Variable(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]]
        )
        f = BLayer(d, BInitializer(d, c))

        y = f(x).data
        self.assertAlmostEqual(1.0, y[0])
        self.assertAlmostEqual(3.0, y[1])
        self.assertAlmostEqual(5.0, y[2])
        self.assertAlmostEqual(7.0, y[3])
        self.assertAlmostEqual(9.0, y[4])

        with tf.GradientTape() as t:
            y = f(x)
        g = t.gradient(y, x).data
        self.assertAlmostEqual(6.0, g[0, 0])
        self.assertAlmostEqual(6.0, g[0, 1])
        self.assertAlmostEqual(6.0, g[0, 2])
        self.assertAlmostEqual(6.0, g[0, 3])
        self.assertAlmostEqual(6.0, g[0, 4])
        self.assertAlmostEqual(2.0, g[1, 0])
        self.assertAlmostEqual(2.0, g[1, 1])
        self.assertAlmostEqual(2.0, g[1, 2])
        self.assertAlmostEqual(2.0, g[1, 3])
        self.assertAlmostEqual(2.0, g[1, 4])


if __name__ == "__main__":
    unittest.main()

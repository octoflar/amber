#  Copyright (c) 2022. Ralf Quast
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
"""
Amber provides code to evaluate multivariate Bernstein polynomials in
TensorFlow using de Casteljau's algorithm. Multivariate Bernstein basis
polynomials are particularly useful for linear multivariate regression
with linear inequality constraints.
"""

__version__ = "2024.0.0"
"""The software version."""


def _tf_config(numpy_behaviour: bool):
    """Configures ``tensorflow``.

    :param: numpy_behaviour Enables experimental numpy behavior, if ``True``.
    """
    try:
        import tensorflow as tf

        if numpy_behaviour:
            tf.experimental.numpy.experimental_enable_numpy_behavior()
    except ModuleNotFoundError:
        pass


_tf_config(numpy_behaviour=True)

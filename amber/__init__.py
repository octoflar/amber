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
from typing import Literal

__version__ = "2024.0.0"
"""The software version."""


def _tf_config(
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"
    ] = "ERROR",
    num_threads: int = 1,
    numpy_behaviour: bool = True,
):
    """Configures ``tensorflow``.

    :param: log_level The log level.
    :param: num_threads The number of threads used by independent non-blocking
    operations. If ``0`` the system picks an appropriate number.
    :param: numpy_behaviour Enables or disables experimental numpy behavior.
    """
    try:
        import logging

        logging.getLogger("absl").setLevel(log_level)
        logging.getLogger("tensorflow").setLevel(log_level)

        import tensorflow as tf

        tf.get_logger().setLevel(log_level)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        if numpy_behaviour:
            tf.experimental.numpy.experimental_enable_numpy_behavior()
    except ModuleNotFoundError:
        pass


_tf_config()

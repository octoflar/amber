![Graphical abstract](assets/img/bernstein-series.png "Series of Bernstein basis polynomials")

Amber provides code to evaluate multivariate Bernstein polynomials using [De Casteljau's
algorithm](https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm). Bernstein polynomials
are useful for, e.g., linear multivariate regression with linear inequality constraints. 

# Getting started [![CMake](https://github.com/octoflar/amber/actions/workflows/cmake.yml/badge.svg)](https://github.com/octoflar/amber/actions/workflows/cmake.yml)
 
Building this software requires [CMake](https://cmake.org) and a compiler that implements
the Fortran 2008 standard. To build the code and run the tests `cd` into the project root
directory and type:

    mkdir cmake-build
    cd cmake-build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make all test

To use a specific Fortran compiler set the `FC` and `CC` environment variables, like

    export FC=gfortran
    export CC=gcc

*before* you execute the `cmake ...` command.

# Further reading

Esmeralda Mainar, J.M. Pena (2006). "Evaluation algorithms for multivariate polynomials in Bernstein–Bezier form."
Journal of Approximation Theory 143, 44–61. <https://doi.org/10.1016/j.jat.2006.05.007>.

Charles L. Lawson and Richard J. Hanson (1995). "Solving Least Squares Problems."
<https://doi.org/10.1137/1.9781611971217>.
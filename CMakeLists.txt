## @author Ralf Quast
## @date 2025
## @copyright MIT License
cmake_minimum_required(VERSION 3.10.0)
include(etc/main/cmake/test.cmake)
include(etc/main/cmake/unix.cmake)
include(etc/main/cmake/openmp.cmake)

##
# Project
#
project(amber VERSION 2025.1 LANGUAGES Fortran)

openmp_optional()

##
# Source directories
#
set(MAIN ${CMAKE_SOURCE_DIR}/etc/main)
set(TEST ${CMAKE_SOURCE_DIR}/etc/test)


##
# Configuration
#
# None.

##
# Main targets
#
# None.


##
# Test targets
#
add_unit_test(bernstein_test
        ${MAIN}/fortran/base_mod.F90
        ${MAIN}/fortran/bernstein_mod.F90
        ${TEST}/fortran/bernstein_mod_test.F90)
add_unit_test(nnls_test
        ${MAIN}/fortran/base_mod.F90
        ${MAIN}/fortran/bernstein_mod.F90
        ${MAIN}/fortran/nnls_mod.F90
        ${TEST}/fortran/nnls_mod_test.F90)


##
# Custom targets
#
add_custom_target(tar
        COMMAND ${TAR} -f ${PROJECT_NAME}.tar.gz -C ${CMAKE_SOURCE_DIR} --exclude .git -X ${CMAKE_SOURCE_DIR}/.gitignore .
        BYPRODUCTS ${PROJECT_NAME}.tar.gz
        VERBATIM)


##
# Installation
#
# None.

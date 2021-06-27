## @author Ralf Quast
## @date 2021
## @copyright MIT License

enable_language(Fortran)

if (${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wuninitialized")
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -pedantic -Og -fcheck=all -finit-real=snan -ffpe-trap=invalid,zero,overflow -fbacktrace")
endif ()

if (${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -gen-interfaces -warn interfaces")
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -traceback -check all -ftrapuv -debug all -fpe3 -fpe-all=3")
endif ()

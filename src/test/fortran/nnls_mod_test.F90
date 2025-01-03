!> @file nnls_mod_test.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
module nnls_mod_testsuite
  use test_mod
  use base_mod
  use bernstein_mod
  use nnls_mod

  implicit none
  private

  public run_testsuite

contains

  !> @brief Directives run once, before all tests.
  subroutine before_all
    call test_initialize
  end subroutine before_all

  !> @brief Directives run once, after all tests.
  subroutine after_all

  end subroutine after_all

  !> @brief Directives run before each test.
  subroutine before

  end subroutine before

  !> @brief Directives run after each test.
  subroutine after

  end subroutine after

  subroutine test_nnls_solve_01
    character(len=*), parameter :: TEST = "test_nnls_solve_01"

    integer,       parameter :: m = 3
    integer,       parameter :: n = 6
    real(kind=dp), parameter :: x(n) = (/ 0.0_dp, 0.2_dp, 0.4_dp, 0.6_dp, 0.8_dp, 1.0_dp /)
    real(kind=dp), parameter :: y(n) = (/ 3.0_dp, 3.0_dp, 3.0_dp, 3.0_dp, 3.0_dp, 3.0_dp /)  ! y = 3
    real(kind=dp)            :: a(m,n)
    real(kind=dp)            :: b(m)
    integer                  :: info

    call bernstein_eval_basis( n, x, m - 1, a )

    call nnls_solve( m, n, a, y, b, info )
    call assert_equals( TEST, 3.0_dp, b(1), 1.0E-6_dp )
    call assert_equals( TEST, 3.0_dp, b(2), 1.0E-6_dp )
    call assert_equals( TEST, 3.0_dp, b(3), 1.0E-6_dp )
    call assert_equals( TEST, 0, info )
  end subroutine test_nnls_solve_01

  subroutine test_nnls_solve_02
    character(len=*), parameter :: TEST = "test_nnls_solve_02"

    integer,       parameter :: m = 3
    integer,       parameter :: n = 6
    real(kind=dp), parameter :: x(n) = (/ 0.0_dp, 0.2_dp, 0.4_dp, 0.6_dp, 0.8_dp, 1.0_dp /)
    real(kind=dp), parameter :: y(n) = (/ 3.0_dp, 3.4_dp, 3.8_dp, 4.2_dp, 4.6_dp, 5.0_dp /)  ! y = 2 x + 3
    real(kind=dp)            :: a(m,n)
    real(kind=dp)            :: b(m)
    integer                  :: info

    call bernstein_eval_basis( n, x, m - 1, a )

    call nnls_solve( m, n, a, y, b, info )
    call assert_equals( TEST, 3.0_dp, b(1), 1.0E-6_dp )
    call assert_equals( TEST, 4.0_dp, b(2), 1.0E-6_dp )
    call assert_equals( TEST, 5.0_dp, b(3), 1.0E-6_dp )
    call assert_equals( TEST, 0, info )
  end subroutine test_nnls_solve_02

  subroutine test_nnls_solve_03
    character(len=*), parameter :: TEST = "test_nnls_solve_03"

    integer,       parameter :: m = 3
    integer,       parameter :: n = 6
    real(kind=dp), parameter :: x(n) = (/ 0.00_dp, 0.20_dp, 0.40_dp, 0.60_dp, 0.80_dp, 1.00_dp /)
    real(kind=dp), parameter :: y(n) = (/ 3.00_dp, 3.44_dp, 3.96_dp, 4.56_dp, 5.24_dp, 6.00_dp /)  ! y = x**2 + 2 x + 3
    real(kind=dp)            :: a(m,n)
    real(kind=dp)            :: b(m)
    integer                  :: info

    call bernstein_eval_basis( n, x, m - 1, a )

    call nnls_solve( m, n, a, y, b, info )
    call assert_equals( TEST, 3.0_dp, b(1), 1.0E-6_dp )
    call assert_equals( TEST, 4.0_dp, b(2), 1.0E-6_dp )
    call assert_equals( TEST, 6.0_dp, b(3), 1.0E-6_dp )
    call assert_equals( TEST, 0, info )
  end subroutine test_nnls_solve_03

  subroutine test_nnls_solve_04
    character(len=*), parameter :: TEST = "test_nnls_solve_04"

    integer,       parameter :: m = 6
    integer,       parameter :: n = 6
    real(kind=dp), parameter :: x(n) = (/ 0.00_dp, 0.20_dp, 0.40_dp, 0.60_dp, 0.80_dp, 1.00_dp /)
    real(kind=dp), parameter :: y(n) = (/ 3.00_dp, 3.44_dp, 3.96_dp, 4.56_dp, 5.24_dp, 6.00_dp /)  ! y = x**2 + 2 x + 3
    real(kind=dp)            :: a(m,n)
    real(kind=dp)            :: b(m)
    integer                  :: info

    call bernstein_eval_basis( n, x, m - 1, a )

    call nnls_solve( m, n, a, y, b, info )
    call assert_equals( TEST, 3.0_dp, b(1), 1.0E-6_dp )
    call assert_equals( TEST, 3.4_dp, b(2), 1.0E-6_dp )
    call assert_equals( TEST, 3.9_dp, b(3), 1.0E-6_dp )
    call assert_equals( TEST, 4.5_dp, b(4), 1.0E-6_dp )
    call assert_equals( TEST, 5.2_dp, b(5), 1.0E-6_dp )
    call assert_equals( TEST, 6.0_dp, b(6), 1.0E-6_dp )
    call assert_equals( TEST, 0, info )
  end subroutine test_nnls_solve_04

  !> @brief Test freely adapted from original NNLS sources.
  subroutine test_nnls_solve_05
    character(len=*), parameter :: TEST = "test_nnls_solve_05"

    integer,       parameter :: m = 19
    integer,       parameter :: n = 100
    real(kind=dp)            :: a(m,n)
    real(kind=dp)            :: x(m)
    real(kind=dp)            :: y(n)
    integer                  :: info
    integer                  :: i, j
    real(kind=dp)            :: t, t0(m)

    t0(1) = 2.0_dp
    t0(2) = t0(1) * sqrt( 2.0_dp )
    do i = 3, m
      t0(i) = 2.0_dp * t0(i-2)
    end do
    do i = 1, n
      t = 10.0_dp * i
      do j = 1, m
        a(j,i) = exp( -t / t0(j) )
      end do
    end do
    do i = 1, n
      t = 10.0_dp * i
      y(i) = 100.0_dp * (exp( -t / 8.0_dp ) + exp( -t / (32.0_dp * sqrt( 2.0_dp)) ) + exp( -t / 512.0_dp ))
    end do

    call nnls_solve( m, n, a, y, x, info )
    call assert_equals( TEST, 0.0_dp,   x(1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(4), 1.0E-6_dp )
    call assert_equals( TEST, 100.0_dp, x(5), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(6), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(7), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(8), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(9), 1.0E-6_dp )
    call assert_equals( TEST, 100.0_dp, x(10), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(11), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(12), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(13), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(14), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(15), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(16), 1.0E-6_dp )
    call assert_equals( TEST, 100.0_dp, x(17), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(18), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,   x(19), 1.0E-6_dp )
    call assert_equals( TEST, 0, info )
  end subroutine test_nnls_solve_05

  !! @brief Add all your test cases here.
  subroutine run_all
    call run( test_nnls_solve_01 )
    call run( test_nnls_solve_02 )
    call run( test_nnls_solve_03 )
    call run( test_nnls_solve_04 )
    call run( test_nnls_solve_05 )
  end subroutine run_all

  !> @brief Runs a test case.
  subroutine run( test )
    interface
      subroutine test
      end subroutine test
    end interface

    call before
    call test
    call after
  end subroutine run

  !> @brief Runs the testsuite.
  subroutine run_testsuite
    call before_all
    call run_all
    call after_all
  end subroutine run_testsuite

end module nnls_mod_testsuite


!> @brief Runs the testsuite.
program main
  use nnls_mod_testsuite, only: run_testsuite

  call run_testsuite
end program main


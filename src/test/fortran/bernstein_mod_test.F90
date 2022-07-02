!> @file bernstein_mod_test.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
module bernstein_mod_testsuite
  use test_mod
  use base_mod
  use bernstein_mod

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

  subroutine test_bernstein_eval_basis_01
    implicit none
    character(len=*), parameter :: TEST = "test_bernstein_eval_basis_01"

    integer, parameter :: m = 3
    integer, parameter :: d = 4
    real(kind=dp),    parameter :: x(m) = (/ 0.0_dp, 0.5_dp, 1.0_dp /)
    real(kind=dp)               :: y(0:d,m)

    call bernstein_eval_basis( m, x, d, y )
    call assert_equals( TEST, 1.0_dp,    y(0,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(1,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(4,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0625_dp, y(0,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.25_dp,   y(1,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.375_dp,  y(2,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.25_dp,   y(3,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0625_dp, y(4,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(0,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(1,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,3), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp,    y(4,3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_basis_01

  subroutine test_bernstein_eval_basis_n_01
    implicit none
    character(len=*), parameter :: TEST = "test_bernstein_eval_basis_n_01"

    integer, parameter :: n = 1
    integer, parameter :: m = 3
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: x(n,m) = reshape( (/ 0.0_dp, 0.5_dp, 1.0_dp /), (/ n, m /) )
    real(kind=dp)               :: y(o,m)

    call bernstein_eval_basis_n( n, m, x, d, y )
    call assert_equals( TEST, 1.0_dp,    y(1,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(4,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(5,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0625_dp, y(1,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.25_dp,   y(2,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.375_dp,  y(3,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.25_dp,   y(4,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0625_dp, y(5,2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(1,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(4,3), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp,    y(5,3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_basis_n_01


  subroutine test_bernstein_eval_basis_n_02
    character(len=*), parameter :: TEST = "test_bernstein_eval_basis_n_02"

    integer, parameter :: n = 2
    integer, parameter :: m = 3
    integer, parameter :: d(n) = (/ 4, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(o,m)

    x(:,1) = (/ 0.0_dp, 0.0_dp /)
    x(:,2) = (/ 0.5_dp, 0.5_dp /)
    x(:,3) = (/ 1.0_dp, 1.0_dp /)

    call bernstein_eval_basis_n( n, m, x, d, y )
    call assert_equals( TEST, 1.0_dp,    y(1,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(4,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(5,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(6,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(7,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(8,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(9,1),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(10,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(11,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(12,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(13,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(14,1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(15,1), 1.0E-6_dp )

    call assert_equals( TEST, 0.015625_dp, y(1,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.0625_dp,   y(2,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.09375_dp,  y(3,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.0625_dp,   y(4,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.015625_dp, y(5,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.03125_dp,  y(6,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.125_dp,    y(7,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.1875_dp,   y(8,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.125_dp,    y(9,2),  1.0E-7_dp )
    call assert_equals( TEST, 0.03125_dp,  y(10,2), 1.0E-7_dp )
    call assert_equals( TEST, 0.015625_dp, y(11,2), 1.0E-7_dp )
    call assert_equals( TEST, 0.0625_dp,   y(12,2), 1.0E-7_dp )
    call assert_equals( TEST, 0.09375_dp,  y(13,2), 1.0E-7_dp )
    call assert_equals( TEST, 0.0625_dp,   y(14,2), 1.0E-7_dp )
    call assert_equals( TEST, 0.015625_dp, y(15,2), 1.0E-7_dp )

    call assert_equals( TEST, 0.0_dp,    y(1,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(2,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(3,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(4,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(5,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(6,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(7,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(8,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(9,3),  1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(10,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(11,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(12,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(13,3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp,    y(14,3), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp,    y(15,3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_basis_n_02

  subroutine test_bernstein_eval_poly_01
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_01"

    integer, parameter :: m = 3
    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = 0.0_dp
    real(kind=dp),    parameter :: x(m) = (/ 0.0_dp, 0.5_dp, 1.0_dp /)
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly( m, x, d, c, y )
    call assert_equals( TEST, 0.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_01

  subroutine test_bernstein_eval_poly_02
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_02"

    integer, parameter :: m = 3
    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = 1.0_dp
    real(kind=dp),    parameter :: x(m) = (/ 0.0_dp, 0.5_dp, 1.0_dp /)
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly( m, x, d, c, y )
    call assert_equals( TEST, 1.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_02

  subroutine test_bernstein_eval_poly_03
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_03"

    integer, parameter :: m = 3
    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = (/ 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp /)
    real(kind=dp),    parameter :: x(m) = (/ 0.3141_dp, 0.2718_dp, 0.5772_dp /)
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly( m, x, d, c, y )
    call assert_equals( TEST, 2.2564_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 2.0872_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 3.3088_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_03

  subroutine test_bernstein_eval_poly_1_01
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_1_01"

    integer, parameter :: n = 1
    integer, parameter :: m = 3
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp),    parameter :: x(n,m) = reshape( (/ 0.0_dp, 0.5_dp, 1.0_dp /), (/ n, m /) )
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 0.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_1_01

  subroutine test_bernstein_eval_poly_1_02
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_1_02"

    integer, parameter :: n = 1
    integer, parameter :: m = 3
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp),    parameter :: x(n,m) = reshape( (/ 0.0_dp, 0.5_dp, 1.0_dp /), (/ n, m /) )
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 1.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_1_02

  subroutine test_bernstein_eval_poly_1_03
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_1_03"

    integer, parameter :: n = 1
    integer, parameter :: m = 3
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = (/ 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp /)
    real(kind=dp),    parameter :: x(n,m) = reshape( (/ 0.3141_dp, 0.2718_dp, 0.5772_dp /), (/ n, m /) )
    real(kind=dp)               :: y(m)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 2.2564_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 2.0872_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 3.3088_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_1_03

  subroutine test_bernstein_eval_poly_2_01
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_2_01"

    integer, parameter :: n = 2
    integer, parameter :: m = 5
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    x(:,1) = (/ 0.0_dp, 0.0_dp /)
    x(:,2) = (/ 1.0_dp, 0.0_dp /)
    x(:,3) = (/ 0.0_dp, 1.0_dp /)
    x(:,4) = (/ 1.0_dp, 1.0_dp /)
    x(:,5) = (/ 0.5_dp, 0.5_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 0.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(4), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(5), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_2_01

  subroutine test_bernstein_eval_poly_2_02
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_2_02"

    integer, parameter :: n = 2
    integer, parameter :: m = 5
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    x(:,1) = (/ 0.0_dp, 0.0_dp /)
    x(:,2) = (/ 1.0_dp, 0.0_dp /)
    x(:,3) = (/ 0.0_dp, 1.0_dp /)
    x(:,4) = (/ 1.0_dp, 1.0_dp /)
    x(:,5) = (/ 0.5_dp, 0.5_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 1.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(3), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(4), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(5), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_2_02

  subroutine test_bernstein_eval_poly_2_03
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_2_03"

    integer, parameter :: n = 2
    integer, parameter :: m = 2
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp)               :: c(o)
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    call fill__dp( n, d, c )
    x(:,1) = (/ 0.2718_dp, 0.5772_dp /)
    x(:,2) = (/ 0.5772_dp, 0.2718_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST,  7.0804_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 11.0506_dp, y(2), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_2_03

  subroutine test_bernstein_eval_poly_3_01
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_3_01"

    integer, parameter :: n = 3
    integer, parameter :: m = 9
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    x(:,1) = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
    x(:,2) = (/ 1.0_dp, 0.0_dp, 0.0_dp /)
    x(:,3) = (/ 0.0_dp, 1.0_dp, 0.0_dp /)
    x(:,4) = (/ 0.0_dp, 0.0_dp, 1.0_dp /)
    x(:,5) = (/ 1.0_dp, 1.0_dp, 0.0_dp /)
    x(:,6) = (/ 1.0_dp, 0.0_dp, 1.0_dp /)
    x(:,7) = (/ 0.0_dp, 1.0_dp, 1.0_dp /)
    x(:,8) = (/ 1.0_dp, 1.0_dp, 1.0_dp /)
    x(:,9) = (/ 0.5_dp, 0.5_dp, 0.5_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 0.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(3), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(4), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(5), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(6), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(7), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(8), 1.0E-6_dp )
    call assert_equals( TEST, 0.0_dp, y(9), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_3_01

  subroutine test_bernstein_eval_poly_3_02
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_3_02"

    integer, parameter :: n = 3
    integer, parameter :: m = 9
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    x(:,1) = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
    x(:,2) = (/ 1.0_dp, 0.0_dp, 0.0_dp /)
    x(:,3) = (/ 0.0_dp, 1.0_dp, 0.0_dp /)
    x(:,4) = (/ 0.0_dp, 0.0_dp, 1.0_dp /)
    x(:,5) = (/ 1.0_dp, 1.0_dp, 0.0_dp /)
    x(:,6) = (/ 1.0_dp, 0.0_dp, 1.0_dp /)
    x(:,7) = (/ 0.0_dp, 1.0_dp, 1.0_dp /)
    x(:,8) = (/ 1.0_dp, 1.0_dp, 1.0_dp /)
    x(:,9) = (/ 0.5_dp, 0.5_dp, 0.5_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 1.0_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(3), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(4), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(5), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(6), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(7), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(8), 1.0E-6_dp )
    call assert_equals( TEST, 1.0_dp, y(9), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_3_02

  subroutine test_bernstein_eval_poly_3_03
    character(len=*), parameter :: TEST = "test_bernstein_eval_poly_3_03"

    integer, parameter :: n = 3
    integer, parameter :: m = 3
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp)               :: c(o)
    real(kind=dp)               :: x(n,m)
    real(kind=dp)               :: y(m)

    call fill__dp( n, d, c )
    x(:,1) = (/ 0.2718_dp, 0.5772_dp, 0.3141_dp /)
    x(:,2) = (/ 0.5772_dp, 0.3141_dp, 0.2718_dp /)
    x(:,3) = (/ 0.3141_dp, 0.2718_dp, 0.5772_dp /)

    call bernstein_eval_poly_n( n, m, x, d, c, y )
    call assert_equals( TEST, 19.8694_dp, y(1), 1.0E-6_dp )
    call assert_equals( TEST, 32.0761_dp, y(2), 1.0E-6_dp )
    call assert_equals( TEST, 19.6774_dp, y(3), 1.0E-6_dp )
  end subroutine test_bernstein_eval_poly_3_03

  subroutine test_bernstein_incr_degree_01
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_01"

    integer, parameter :: d = 4
    integer, parameter :: incr = 1
    real(kind=dp)      :: b(0:d)
    real(kind=dp)      :: c(0:d + incr)
    real(kind=dp)      :: x

    b(0) = 1.0_dp
    b(1) = 2.0_dp
    b(2) = 3.0_dp
    b(3) = 4.0_dp
    b(4) = 5.0_dp

    call bernstein_incr_degree( incr, d, b, c )

    x = 0.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.2718_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.5772_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 5.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_01

  subroutine test_bernstein_incr_degree_02
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_02"

    integer, parameter :: d = 4
    integer, parameter :: incr = 2
    real(kind=dp)      :: b(0:d)
    real(kind=dp)      :: c(0:d + incr)
    real(kind=dp)      :: x

    b(0) = 1.0_dp
    b(1) = 2.0_dp
    b(2) = 3.0_dp
    b(3) = 4.0_dp
    b(4) = 5.0_dp

    call bernstein_incr_degree( incr, d, b, c )

    x = 0.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.2718_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.5772_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 5.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_02

  subroutine test_bernstein_incr_degree_03
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_03"

    integer, parameter :: d = 4
    integer, parameter :: incr = 3
    real(kind=dp)      :: b(0:d)
    real(kind=dp)      :: c(0:d + incr)
    real(kind=dp)      :: x

    b(0) = 1.0_dp
    b(1) = 2.0_dp
    b(2) = 3.0_dp
    b(3) = 4.0_dp
    b(4) = 5.0_dp

    call bernstein_incr_degree( incr, d, b, c )

    x = 0.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.2718_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 0.5772_dp
    call assert_equals( TEST, bernstein_poly( x, d, b ), bernstein_poly( x, d + incr, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 5.0_dp, bernstein_poly( x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_03

  subroutine test_bernstein_incr_degree_n_01
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_n_01"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 3, 2 /)
    integer, parameter :: incr(n) = (/ 0, 1 /)
    real(kind=dp)      :: b(12)
    real(kind=dp)      :: c(16)
    real(kind=dp)      :: x(n)

    b(1) = 1.0_dp
    b(2) = 2.0_dp
    b(3) = 3.0_dp
    b(4) = 4.0_dp
    b(5) = 5.0_dp
    b(6) = 6.0_dp
    b(7) = 7.0_dp
    b(8) = 8.0_dp
    b(9) = 9.0_dp
    b(10) = 10.0_dp
    b(11) = 11.0_dp
    b(12) = 12.0_dp

    call bernstein_incr_degree_n( n, incr, d, b, c )

    x = (/ 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.2718_dp, 0.3141_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.3141_dp, 0.5772_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 12.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_n_01

  subroutine test_bernstein_incr_degree_n_02
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_n_02"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 3, 2 /)
    integer, parameter :: incr(n) = (/ 1, 0 /)
    real(kind=dp)      :: b(12)
    real(kind=dp)      :: c(15)
    real(kind=dp)      :: x(n)

    b(1) = 1.0_dp
    b(2) = 2.0_dp
    b(3) = 3.0_dp
    b(4) = 4.0_dp
    b(5) = 5.0_dp
    b(6) = 6.0_dp
    b(7) = 7.0_dp
    b(8) = 8.0_dp
    b(9) = 9.0_dp
    b(10) = 10.0_dp
    b(11) = 11.0_dp
    b(12) = 12.0_dp

    call bernstein_incr_degree_n( n, incr, d, b, c )

    x = (/ 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.2718_dp, 0.3141_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.3141_dp, 0.5772_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 12.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_n_02

  subroutine test_bernstein_incr_degree_n_03
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_n_03"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 3, 2 /)
    integer, parameter :: incr(n) = (/ 2, 4 /)
    real(kind=dp)      :: b(12)
    real(kind=dp)      :: c(42)
    real(kind=dp)      :: x(n)

    b(1) = 1.0_dp
    b(2) = 2.0_dp
    b(3) = 3.0_dp
    b(4) = 4.0_dp
    b(5) = 5.0_dp
    b(6) = 6.0_dp
    b(7) = 7.0_dp
    b(8) = 8.0_dp
    b(9) = 9.0_dp
    b(10) = 10.0_dp
    b(11) = 11.0_dp
    b(12) = 12.0_dp

    call bernstein_incr_degree_n( n, incr, d, b, c )

    x = (/ 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.2718_dp, 0.3141_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.3141_dp, 0.5772_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 12.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_n_03

  subroutine test_bernstein_incr_degree_n_04
    character(len=*), parameter :: TEST = "test_bernstein_incr_degree_n_04"

    integer, parameter :: n = 3
    integer, parameter :: d(n) = (/ 1, 1, 2 /)
    integer, parameter :: incr(n) = (/ 2, 4, 1 /)
    real(kind=dp)      :: b(12)
    real(kind=dp)      :: c(96)
    real(kind=dp)      :: x(n)

    b(1) = 1.0_dp
    b(2) = 2.0_dp
    b(3) = 3.0_dp
    b(4) = 4.0_dp
    b(5) = 5.0_dp
    b(6) = 6.0_dp
    b(7) = 7.0_dp
    b(8) = 8.0_dp
    b(9) = 9.0_dp
    b(10) = 10.0_dp
    b(11) = 11.0_dp
    b(12) = 12.0_dp

    call bernstein_incr_degree_n( n, incr, d, b, c )

    x = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.2718_dp, 0.3141_dp, 0.7071_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 0.3141_dp, 0.7071_dp, 0.5772_dp /)
    call assert_equals( TEST, bernstein_poly_n( n, x, d, b ), bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 12.0_dp, bernstein_poly_n( n, x, d + incr, c ), 1.0E-6_dp )
  end subroutine test_bernstein_incr_degree_n_04

  subroutine test_bernstein_poly_01
    character(len=*), parameter :: TEST = "test_bernstein_poly_01"

    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = 0.0_dp
    real(kind=dp)               :: x

    x = 0.0_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )

    x = 0.5_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_01

  subroutine test_bernstein_poly_02
    character(len=*), parameter :: TEST = "test_bernstein_poly_02"

    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = 1.0_dp
    real(kind=dp)               :: x

    x = 0.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )

    x = 0.5_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly( x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_02

  subroutine test_bernstein_poly_03
    character(len=*), parameter :: TEST = "test_bernstein_poly_03"

    integer, parameter :: d = 4
    real(kind=dp),    parameter :: c(d + 1) = (/ 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp /)

    call assert_equals( TEST, 2.2564_dp, bernstein_poly( 0.3141_dp, d, c ), 1.0E-6_dp )
    call assert_equals( TEST, 2.0872_dp, bernstein_poly( 0.2718_dp, d, c ), 1.0E-6_dp )
    call assert_equals( TEST, 3.3088_dp, bernstein_poly( 0.5772_dp, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_03

  subroutine test_bernstein_poly_1_01
    character(len=*), parameter :: TEST = "test_bernstein_poly_1_01"

    integer, parameter :: n = 1
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp)               :: x(n)

    x = 0.0_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 0.5_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_1_01

  subroutine test_bernstein_poly_1_02
    character(len=*), parameter :: TEST = "test_bernstein_poly_1_02"

    integer, parameter :: n = 1
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp)               :: x(n)

    x = 0.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 0.5_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 1.0_dp
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_1_02

  subroutine test_bernstein_poly_1_03
    character(len=*), parameter :: TEST = "test_bernstein_poly_1_03"

    integer, parameter :: n = 1
    integer, parameter :: d(n) = 4
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = (/ 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp /)
    real(kind=dp)               :: x(n)

    x = 0.3141_dp
    call assert_equals( TEST, 2.2564_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 0.2718_dp
    call assert_equals( TEST, 2.0872_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = 0.5772_dp
    call assert_equals( TEST, 3.3088_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_1_03

  subroutine test_bernstein_poly_2_01
    character(len=*), parameter :: TEST = "test_bernstein_poly_2_01"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp)               :: x(n)

    x = (/ 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5_dp, 0.5_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_2_01

  subroutine test_bernstein_poly_2_02
    character(len=*), parameter :: TEST = "test_bernstein_poly_2_02"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp)               :: x(n)

    x = (/ 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5_dp, 0.5_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_2_02

  subroutine test_bernstein_poly_2_03
    character(len=*), parameter :: TEST = "test_bernstein_poly_2_03"

    integer, parameter :: n = 2
    integer, parameter :: d(n) = (/ 4, 3 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp)               :: c(o)
    real(kind=dp)               :: x(n)

    call fill__dp( n, d, c )

    x = (/ 0.2718_dp, 0.5772_dp /)
    call assert_equals( TEST,  7.0804_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5772_dp, 0.2718_dp /)
    call assert_equals( TEST, 11.0506_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_2_03

  subroutine test_bernstein_poly_3_01
    character(len=*), parameter :: TEST = "test_bernstein_poly_3_01"

    integer, parameter :: n = 3
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 0.0_dp
    real(kind=dp)               :: x(n)

    x = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5_dp, 0.5_dp, 0.5_dp /)
    call assert_equals( TEST, 0.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_3_01

  subroutine test_bernstein_poly_3_02
    character(len=*), parameter :: TEST = "test_bernstein_poly_3_02"

    integer, parameter :: n = 3
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp),    parameter :: c(o) = 1.0_dp
    real(kind=dp)               :: x(n)

    x = (/ 0.0_dp, 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp, 0.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 0.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.0_dp, 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 1.0_dp, 1.0_dp, 1.0_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5_dp, 0.5_dp, 0.5_dp /)
    call assert_equals( TEST, 1.0_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_3_02

  subroutine test_bernstein_poly_3_03
    character(len=*), parameter :: TEST = "test_bernstein_poly_3_03"

    integer, parameter :: n = 3
    integer, parameter :: d(n) = (/ 4, 3, 2 /)
    integer, parameter :: o = product( d + 1 )
    real(kind=dp)               :: c(o)
    real(kind=dp)               :: x(n)

    call fill__dp( n, d, c )

    x = (/ 0.2718_dp, 0.5772_dp, 0.3141_dp /)
    call assert_equals( TEST, 19.8694_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.5772_dp, 0.3141_dp, 0.2718_dp /)
    call assert_equals( TEST, 32.0761_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )

    x = (/ 0.3141_dp, 0.2718_dp, 0.5772_dp /)
    call assert_equals( TEST, 19.6774_dp, bernstein_poly_n( n, x, d, c ), 1.0E-6_dp )
  end subroutine test_bernstein_poly_3_03

  !> @brief Fills an n-variate Bernstein batch with natural numbers 1, 2, 3, ...
  !> @param n[in] The dimension.
  !> @param d[in] The degrees of the Bernstein batch.
  !> @param b[out] The filled Bernstein batch.
  !> @details The memory layout of the elements in the Bernstein batch is the same as the memory
  !! layout of a multidimensional array with dimensions b(0:d(1),0:d(2),...,0:d(n))
  pure subroutine fill__dp( n, d, b )
    integer,       intent(in)  :: n
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(out) :: b(product( d + 1 ))
    integer                    :: strides(n)
    integer                    :: indexes(n)
    integer                    :: address
    integer                    :: i, k

    ! Compute the stride in each dimension. The stride in the first dimension is always 1
    strides = 1
    do i = 1, n - 1
      strides(i + 1) = strides(i) * (d(i) + 1)
    end do
    ! Set the multi-index to the first element in the Bernstein batch
    indexes = 0
    ! Iterate over the Bernstein batch
    do k = 1, size( b )
      ! Compute the 'address' of the k-th element in the Bernstein batch
      address = dot_product( strides, indexes ) + 1
      ! Set k-th element in the Bernstein batch to k.
      b(address) = k
      ! Increment the multi-index
      do i = n, 1, -1
        if (indexes(i) < d(i)) then
          indexes(i) = indexes(i) + 1
          exit
        else
          indexes(i) = 0
        end if
      end do
    end do
  end subroutine fill__dp


  !! @brief Add all your test cases here.
  subroutine run_all
    call run( test_bernstein_eval_basis_01 )

    call run( test_bernstein_eval_basis_n_01 )
    call run( test_bernstein_eval_basis_n_02 )

    call run( test_bernstein_eval_poly_01 )
    call run( test_bernstein_eval_poly_02 )
    call run( test_bernstein_eval_poly_03 )

    call run( test_bernstein_eval_poly_1_01 )
    call run( test_bernstein_eval_poly_1_02 )
    call run( test_bernstein_eval_poly_1_03 )

    call run( test_bernstein_eval_poly_2_01 )
    call run( test_bernstein_eval_poly_2_02 )
    call run( test_bernstein_eval_poly_2_03 )

    call run( test_bernstein_eval_poly_3_01 )
    call run( test_bernstein_eval_poly_3_02 )
    call run( test_bernstein_eval_poly_3_03 )

    call run( test_bernstein_incr_degree_01 )
    call run( test_bernstein_incr_degree_02 )
    call run( test_bernstein_incr_degree_03 )

    call run( test_bernstein_incr_degree_n_01 )
    call run( test_bernstein_incr_degree_n_02 )
    call run( test_bernstein_incr_degree_n_03 )
    call run( test_bernstein_incr_degree_n_04 )

    call run( test_bernstein_poly_01 )
    call run( test_bernstein_poly_02 )
    call run( test_bernstein_poly_03 )

    call run( test_bernstein_poly_1_01 )
    call run( test_bernstein_poly_1_02 )
    call run( test_bernstein_poly_1_03 )

    call run( test_bernstein_poly_2_01 )
    call run( test_bernstein_poly_2_02 )
    call run( test_bernstein_poly_2_03 )

    call run( test_bernstein_poly_3_01 )
    call run( test_bernstein_poly_3_02 )
    call run( test_bernstein_poly_3_03 )
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

end module bernstein_mod_testsuite


!> @brief Runs the testsuite.
program main
  use bernstein_mod_testsuite, only: run_testsuite

  call run_testsuite
end program main


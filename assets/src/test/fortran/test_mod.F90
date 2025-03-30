!> @file test_mod.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
!> @brief Module for unit-level testing in the style of [JUnit](http://junit.org/)
module test_mod
  implicit none
  private

  public assert_equals, assert_not_equals
  public assert_true, assert_false
  public assert_nan, assert_not_nan

  public test_initialize

  interface assert_equals
    module procedure assert_equivalent, assert_equals_real__dp, assert_equals_real__sp, assert_equals_int
  end interface

  interface assert_not_equals
    module procedure assert_not_equivalent, assert_not_equals_real__dp, assert_not_equals_real__sp, assert_not_equals_int
  end interface

  interface assert_nan
    module procedure assert_nan__dp, assert_nan__sp
  end interface

  interface assert_not_nan
    module procedure assert_not_nan__dp, assert_not_nan__sp
  end interface

  interface is_nan
    module procedure is_nan__dp, is_nan__sp
  end interface

  logical :: error_stop_on_failure = .true.
  logical :: message_on_success    = .true.
  logical :: message_on_failure    = .true.
  integer :: failure_output_unit   = 0
  integer :: success_output_unit   = 0

  namelist /config/        &
    error_stop_on_failure, &
    message_on_success,    &
    message_on_failure,    &
    failure_output_unit,   &
    success_output_unit

contains

  !> @brief Initializes the module with properties read from 'test.par', if present.
  subroutine test_initialize
    use, intrinsic :: iso_fortran_env, only: out => output_unit, err => error_unit

    logical                     :: ok
    integer,          parameter :: UNIT = 12
    character(len=*), parameter :: CONFIG_FILE = 'test.par'

    inquire( file=CONFIG_FILE, exist=ok )
    if (ok) then
      open ( UNIT, file=CONFIG_FILE )
      read ( UNIT, nml=config )
      close( UNIT )
    end if
    select case (failure_output_unit)
      case (0)
        failure_output_unit = out
      case (1)
        failure_output_unit = err
    end select
    select case (success_output_unit)
      case (0)
        success_output_unit = out
      case (1)
        success_output_unit = err
    end select
  end subroutine test_initialize

  !> @brief Asserts that a logical expression is '.true.'.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value of the logical expression.
  subroutine assert_true( test, actual )
    character(len=*), intent(in) :: test
    logical, intent(in) :: actual

    if (.not. actual) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", .true.
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_true

  !> @brief Asserts that a logical expression is '.false.'.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value of the logical expression.
  subroutine assert_false( test, actual )
    character(len=*), intent(in) :: test
    logical, intent(in) :: actual

    if (actual) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", .false.
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_false

  !> @brief Asserts that two logical expressions are equivalent.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value of the logical expression.
  !> @param[in] actual The actual value of the logical expression.
  subroutine assert_equivalent( test, expected, actual )
    character(len=*), intent(in) :: test
    logical, intent(in) :: expected
    logical, intent(in) :: actual

    if (actual .neqv. expected) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_equivalent

  !> @brief Asserts that two logical expressions are not equivalent.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value of the logical expression.
  !> @param[in] actual The actual value of the logical expression.
  subroutine assert_not_equivalent( test, expected, actual )
    character(len=*), intent(in) :: test
    logical, intent(in) :: expected
    logical, intent(in) :: actual

    if (actual .eqv. expected) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equivalent to ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_equivalent

  !> @brief Asserts that two integral numbers are equal.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  subroutine assert_equals_int( test, expected, actual )
    character(len=*), intent(in) :: test
    integer, intent(in) :: expected
    integer, intent(in) :: actual

    if (actual /= expected) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_equals_int

  !> @brief Asserts that two integral numbers are not equal.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  subroutine assert_not_equals_int( test, expected, actual )
    character(len=*), intent(in) :: test
    integer, intent(in) :: expected
    integer, intent(in) :: actual

    if (actual == expected) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equal to ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_equals_int

  !> @brief Asserts that two real numbers are equal within a given tolerance.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  !> @param[in] tolerance The absolute tolerance.
  subroutine assert_equals_real__dp( test, expected, actual, tolerance )
    use base_mod, only: dp

    character(len=*), intent(in) :: test
    real(kind=dp), intent(in) :: expected
    real(kind=dp), intent(in) :: actual
    real(kind=dp), intent(in) :: tolerance

    if (abs( actual - expected ) > tolerance) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
        write(failure_output_unit, *) "   Expected tolerance: ", tolerance
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_equals_real__dp

  !> @brief Asserts that two real numbers are equal within a given tolerance.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  !> @param[in] tolerance The absolute tolerance.
  subroutine assert_equals_real__sp( test, expected, actual, tolerance )
    use base_mod, only: sp

    character(len=*), intent(in) :: test
    real(kind=sp), intent(in) :: expected
    real(kind=sp), intent(in) :: actual
    real(kind=sp), intent(in) :: tolerance

    if (abs( actual - expected ) > tolerance) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
        write(failure_output_unit, *) "   Expected tolerance: ", tolerance
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_equals_real__sp

  !> @brief Asserts that two real numbers are not equal.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  !> @param[in] tolerance The absolute tolerance.
  subroutine assert_not_equals_real__dp( test, expected, actual, tolerance )
    use base_mod, only: dp

    character(len=*), intent(in) :: test
    real(kind=dp), intent(in) :: expected
    real(kind=dp), intent(in) :: actual
    real(kind=dp), intent(in) :: tolerance

    if (abs( actual - expected ) < tolerance) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equal to ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
        write(failure_output_unit, *) "   Expected tolerance: ", tolerance
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_equals_real__dp

  !> @brief Asserts that two real numbers are not equal.
  !> @param[in] test The name of the caller.
  !> @param[in] expected The expected value.
  !> @param[in] actual The actual value.
  !> @param[in] tolerance The absolute tolerance.
  subroutine assert_not_equals_real__sp( test, expected, actual, tolerance )
    use base_mod, only: sp

    character(len=*), intent(in) :: test
    real(kind=sp), intent(in) :: expected
    real(kind=sp), intent(in) :: actual
    real(kind=sp), intent(in) :: tolerance

    if (abs( actual - expected ) < tolerance) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equal to ", expected
        write(failure_output_unit, *) "   Actual   result: ", actual
        write(failure_output_unit, *) "   Expected tolerance: ", tolerance
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_equals_real__sp

  !> @brief Asserts that a real number is NaN.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value.
  subroutine assert_nan__dp( test, actual )
    use base_mod, only: dp

    character(len=*), intent(in) :: test
    real(kind=dp), intent(in) :: actual

    if (.not. is_nan(actual)) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", "NaN"
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_nan__dp

  !> @brief Asserts that a real number is NaN.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value.
  subroutine assert_nan__sp( test, actual )
    use base_mod, only: sp

    character(len=*), intent(in) :: test
    real(kind=sp), intent(in) :: actual

    if (.not. is_nan(actual)) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: ", "NaN"
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_nan__sp

  !> @brief Asserts that a real number is not NaN.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value.
  subroutine assert_not_nan__dp( test, actual )
    use base_mod, only: dp

    character(len=*), intent(in) :: test
    real(kind=dp), intent(in) :: actual

    if (is_nan(actual)) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equal to ", "NaN"
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_nan__dp

  !> @brief Asserts that a real number is not NaN.
  !> @param[in] test The name of the caller.
  !> @param[in] actual The actual value.
  subroutine assert_not_nan__sp( test, actual )
    use base_mod, only: sp

    character(len=*), intent(in) :: test
    real(kind=sp), intent(in) :: actual

    if (is_nan(actual)) then
      if (message_on_failure) then
        write(failure_output_unit, *) test, ": Assertion failed"
        write(failure_output_unit, *) "   Expected result: not equal to ", "NaN"
        write(failure_output_unit, *) "   Actual   result: ", actual
      end if
      call conditional_error_stop
    else
      call conditional_success_message( test )
    end if
  end subroutine assert_not_nan__sp

  !> @brief Sets the error-stop-on-failure property to a new value.
  !> @param b[in] The new property value.
  subroutine set_error_stop_on_failure( b )
    logical, intent(in) :: b

    error_stop_on_failure = b
  end subroutine set_error_stop_on_failure

  pure function is_nan__dp( x ) result (nan)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    use base_mod, only: dp

    real(kind=dp), intent(in) :: x
    logical                   :: nan

    nan = ieee_is_nan( x )
  end function is_nan__dp

  pure function is_nan__sp( x ) result (nan)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    use base_mod, only: sp

    real(kind=sp), intent(in) :: x
    logical                   :: nan

    nan = ieee_is_nan( x )
  end function is_nan__sp

  subroutine conditional_error_stop
    if (error_stop_on_failure) then
      error stop "Unit test failed"
    end if
  end subroutine

  subroutine conditional_success_message( test )
    character(len=*), intent(in) :: test

    if (message_on_success) then
      write(success_output_unit, *) test, ": Assertion passed"
    end if
  end subroutine conditional_success_message

end module test_mod

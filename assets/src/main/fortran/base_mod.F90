!> @file base_mod.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
!> @brief Numerical constants
module base_mod
  implicit none
  private
  public :: sp, dp, wp, xp

  !> IEEE S_floating ("single precision")
  integer, parameter :: sp = selected_real_kind(p= 6,r= 37)

  !> IEEE T_floating ("double precision")
  integer, parameter :: dp = selected_real_kind(p=15,r=307)

  !> Working precision
  integer, parameter :: wp = dp

  !> Data precision
  integer, parameter :: xp = sp

end module base_mod

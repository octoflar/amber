!> @file base_mod.F90
!> @author Ralf Quast
!> @data 2021
!> @copyright MIT License
!> @brief Numerical constants
module base_mod
  implicit none
  private
  public :: sp, dp, mp, ck

  ! at least IEEE S_floating ("single precision")
  integer, parameter :: sp = selected_real_kind(p= 6,r= 37)

  ! at least IEEE T_floating ("double precision")
  integer, parameter :: dp = selected_real_kind(p=15,r=307)

  ! "my" precision -- use this to set/switch precision globally
  integer, parameter :: mp = dp

  ! precision of coordinates
  integer, parameter :: ck = sp

end module base_mod

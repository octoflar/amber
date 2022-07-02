!> @file nnls_mod.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
!> @brief Module to solve non-negative least squares (NNLS) problems.
!>
!> Charles L. Lawson and Richard J. Hanson (1995). "Solving Least Squares Problems."
!> <https://doi.org/10.1137/1.9781611971217>.
!>
!> Alan Bridle (National Radio Astronomy Observatory, NRAO) on NNLS:
!>
!> "NNLS distinguishes itself on bright, compact sources that neither
!> CLEAN nor MEM can process adequately. [...] both CLEAN and MEM
!> produce artifacts that resemble calibration errors and that limit
!> dynamic range. NNLS has no difficulty imaging such sources. It also
!> has no difficulty with sharp edges, such as those of planets or of
!> strong shocks, and can be very advantageous in producing models for
!> self-calibration for both types of sources. [...] NNLS deconvolution
!> can reach the thermal noise limit in VLBA images for which CLEAN
!> produces demonstrably worse solutions. NNLS is therefore a powerful
!> deconvolution algorithm for making high dynamic range images of
!> compact sources for which strong finite support constraints are
!> applicable."
!>
!> "An interesting thing about NNLS is that it is solved iteratively, but
!> as Lawson and Hanson (1995) show, the iteration always converges and
!> terminates. There is no cutoff in iteration required. Sometimes it might
!> run too long, and have to be terminated, but the solution will still be
!> fairly good, since the solution improves smoothly with iteration. Noise,
!> as expected, increases the number of iterations required to reach the
!> solution."
module nnls_mod
  implicit none
  private

  !> @brief Solves a linear optimization problem (A**T x = y) subject to x >= 0.
  interface nnls_solve
    module procedure nnls_solve__dp
  end interface

  public nnls_solve

contains

  !> @brief Solves a linear optimization problem (A**T x = y) subject to x >= 0.
  !> @param[in] m The number of basis functions.
  !> @param[in] n The number of data points.
  !> @param[in] a The basis functions evaluated at the data points.
  !> @param[in] y The objective function evaluated the data points.
  !> @param[out] x The optimised linear coefficients.
  !> @param info[out] Status information. Equal to zero, if no error has occurred.
  subroutine nnls_solve__dp( m, n, a, y, x, info )
    use base_mod, only: dp

    integer,       intent(in)  :: m
    integer,       intent(in)  :: n
    real(kind=dp), intent(in)  :: a(m,n)
    real(kind=dp), intent(in)  :: y(n)
    real(kind=dp), intent(out) :: x(m)
    integer,       intent(out) :: info

    real(kind=dp)              :: b(n)
    real(kind=dp)              :: c(n,m)
    integer                    :: indx(m)
    integer                    :: mode
    real(kind=dp)              :: rnorm
    real(kind=dp)              :: w(m)

    b = y
    c = transpose( a )
    x = 0.0_dp
    info = 0

    call nnls__dp( c, n, m, b, x, rnorm, w, indx, mode )
    select case (mode)
      case (1)
        info = 0
      case (2)
        info = 1
      case (3)
        info = 2
    end select
  end subroutine nnls_solve__dp

  !> @brief Solves a non-negative linear least squares problem.
  !>
  !> Given an \c m by \c n matrix A and an \c m vector B computes an
  !> \c n vector X that solves the least squares problem A * X = B, subject to X >= 0.
  !>
  !> @authors The original version of this code was developed by
  !> Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
  !> 1973 JUN 15, and published in the book
  !> "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
  !> Revised FEB 1995 to accompany reprinting of the book by SIAM.
  !> The translation into Fortran 90 by Alan Miller, February 1997
  !> Latest revision - 15 April 1997
  !>
  !> @param[inout] a On entry contains the matrix A. On exit contains the
  !>        matrix product Q * A, where Q is an orthogonal \m by \m matrix
  !>        generated implicitly by this subroutine.
  !> @param[in] m The number of rows in the matrix A.
  !> @param[in] n The number of columns in the matrix A.
  !> @param[inout] b On entry contains the \c m vector B. On exit contains Q * B.
  !> @param[out] x The solution vector.
  !> @param[out] rnorm The Euclidean norm of the residual vector.
  !> @param[out] w An \c n array of working space. On exit \c w will contain the dual
  !>        solution vector, which will satisfy \c w(i) \c = \c 0 for all \c i in set P
  !>        and \c w(i) \c <= \c 0 for all \c i in set Z.
  !> @param[out] indx An integer working array of length at least \c n.
  !> @param[out] mode A success-failure flag with the meanings:
  !>        (1) the solution has been computed successfully,
  !>        (2) the dimensions \c m or \c n are negative,
  !>        (3) the maximum number of iterations (\c 3 \c n) has been exceeded.
  subroutine nnls__dp (a, m, n, b, x, rnorm, w, indx, mode)
    use base_mod, only: dp

    integer,       intent(in)    :: m, n
    integer,       intent(out)   :: indx(n), mode
    real(kind=dp), intent(inout) :: a(m,n), b(m)
    real(kind=dp), intent(out)   :: x(n), rnorm, w(n)

    ! LOCAL VARIABLES
    integer       :: i, ii, ip, iter, itmax, iz, iz1, iz2, izmax, j, jj, jz, l, mda, npp1, nsetp
    real(kind=dp) :: zz(m)
    real(kind=dp) :: dummy(1)
    real(kind=dp) :: alpha, asave, cc, sm, ss, t, temp, unorm, up, wmax, ztest
    real(kind=dp), parameter :: factor = 0.01_dp, two = 2.0_dp, zero = 0.0_dp

    mode = 1

    if (m <= 0 .or. n <= 0) then
      mode = 2
      return
    end if

    iter = 0
    itmax = 3 * n

    do i = 1, n
      x(i) = zero
      indx(i) = i
    end do

    iz2 = n
    iz1 = 1
    izmax = 0
    nsetp = 0
    npp1 = 1

    ! MAIN LOOP BEGINS HERE.
    ! QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION OR IF M COLS OF A HAVE BEEN TRIANGULARIZED
30  if (iz1 > iz2 .or. nsetp >= m) go to 350

    ! COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W()
    do iz = iz1, iz2
      j = indx(iz)
      w(j) = dot_product( a(npp1:m,j), b(npp1:m) )
    end do

    ! FIND LARGEST POSITIVE W(J)
60  wmax = zero
    do iz = iz1,iz2
      j = indx(iz)
      if (w(j) > wmax) then
        wmax = w(j)
        izmax = iz
      end if
    end do

    ! IF WMAX <= 0 GO TO TERMINATION. INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS
    if (wmax <= zero) go to 350
    iz = izmax
    j = indx(iz)

    ! THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P.
    ! BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID NEAR LINEAR DEPENDENCE.
    asave = a(npp1,j)
    call h12__dp( 1, npp1, npp1 + 1, m, a(:,j), up, dummy, 1, 1, 0 )
    unorm = zero
    if (nsetp /= 0) then
      unorm = sum( a(1:nsetp,j)**2 )
    end if
    unorm = sqrt( unorm )
    if (unorm + abs( a(npp1,j) ) * factor - unorm > zero) then
      ! COL J IS SUFFICIENTLY INDEPENDENT. COPY B INTO ZZ, UPDATE ZZ AND SOLVE FOR ZTEST (PROPOSED NEW VALUE FOR X(J))
      zz(1:m) = b(1:m)
      call h12__dp( 2, npp1, npp1+1, m, a(:,j), up, zz, 1, 1, 1 )
      ztest = zz(npp1) / a(npp1,j)
      ! SEE IF ZTEST IS POSITIVE
      if (ztest > zero) go to 140
    end if

    ! REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P.
    ! RESTORE A(NPP1,J), SET W(J) = 0., AND LOOP BACK TO TEST DUAL COEFFICIENTS AGAIN
    a(npp1,j) = asave
    w(j) = zero
    go to 60

    ! THE INDEX J = indx(IZ) HAS BEEN SELECTED TO BE MOVED FROM SET Z TO SET P.
    ! UPDATE B, UPDATE INDICES, APPLY HOUSEHOLDER TRANSFORMATIONS TO COLS IN NEW SET Z,
    ! ZERO SUBDIAGONAL ELEMENTS IN COL J, SET W(J) = 0
140 b(1:m) = zz(1:m)

    indx(iz) = indx(iz1)
    indx(iz1) = j
    iz1 = iz1 + 1
    nsetp = npp1
    npp1 = npp1 + 1

    mda = size( a, 1 )
    if (iz1  <=  iz2) then
      do jz = iz1, iz2
        jj = indx(jz)
        call h12__dp(2, nsetp, npp1, m, a(:,j), up, a(:,jj), 1, mda, 1 )
      end do
    end if

    if (nsetp /= m) then
      a(npp1:m,j) = zero
    end if

    w(j) = zero
    ! SOLVE THE TRIANGULAR SYSTEM. STORE THE SOLUTION TEMPORARILY IN ZZ()
    call solve_triangular( zz )

    ! SECONDARY LOOP BEGINS HERE.
    ! ITERATION COUNTER
210 iter = iter + 1
    if (iter > itmax) then
      mode = 3
      go to 350
    end if

    ! SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE. IF NOT COMPUTE ALPHA
    alpha = two
    do ip = 1,nsetp
      l = indx(ip)
      if (zz(ip) <= zero) then
        t = -x(l) / (zz(ip) - x(l))
        if (alpha > t) then
          alpha = t
          jj = ip
        end if
      end if
    end do

    ! IF ALL NEW CONSTRAINED COEFFICIENTSS ARE FEASIBLE THEN ALPHA WILL STILL BE 2.
    ! IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP
    if (alpha == two) go to 330
    ! OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0. AND 1. TO
    ! INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
    do ip = 1, nsetp
      l = indx(ip)
      x(l) = x(l) + alpha * ((ip) - x(l))
    end do

    ! MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I FROM SET P TO SET Z.
    i = indx(jj)
260 x(i) = zero
    if (jj /= nsetp) then
      jj = jj + 1
      do j = jj, nsetp
        ii = indx(j)
        indx(j - 1) = ii
        call g1__dp( a(j-1,ii), a(j,ii), cc, ss, a(j-1,ii) )
        a(j,ii) = zero
        do l = 1, n
          if (l /= ii) then
            ! APPLY PROCEDURE G2( CC, SS, A(J-1,L), A(J,L) )
            temp = a(j-1,l)
            a(j-1,l) =  cc * temp + ss * a(j,l)
            a(j,l)   = -ss * temp + cc * a(j,l)
          end if
        end do
        ! APPLY PROCEDURE G2( CC, SS, B(J-1), B(J) )
        temp = b(j-1)
        b(j-1) =  cc * temp + ss * b(j)
        b(j)   = -ss * temp + cc * b(j)
      end do
    end if

    npp1 = nsetp
    nsetp = nsetp-1
    iz1 = iz1-1
    indx(iz1) = i

    ! SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE. THEY SHOULD
    ! BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
    ! IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY
    ! THAT ARE NONPOSITIVE WILL BE SET TO ZERO
    ! AND MOVED FROM SET P TO SET Z
    do jj = 1, nsetp
      i = indx(jj)
      if (x(i) <= zero) go to 260
    end do

    ! COPY B( ) INTO ZZ( ). THEN SOLVE AGAIN AND LOOP BACK
    zz(1:m) = b(1:m)
    call solve_triangular( zz )
    go to 210
    ! END OF SECONDARY LOOP
    330 do ip = 1, nsetp
      i = indx(ip)
      x(i) = zz(ip)
    end do
    ! ALL NEW COEFFICIENTSS ARE POSITIVE.  LOOP BACK TO BEGINNING
    go to 30

    ! END OF MAIN LOOP

    ! COME TO HERE FOR TERMINATION.
    ! COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR
350 sm = zero
    if (npp1 <= m) then
      sm = sum( b(npp1:m)**2 )
    else
      w(1:n) = zero
    end if
    rnorm = sqrt( sm )
    return

  contains

    ! THE FOLLOWING BLOCK OF CODE IS USED AS AN INTERNAL SUBROUTINE
    ! TO SOLVE THE TRIANGULAR SYSTEM, PUTTING THE SOLUTION IN ZZ().
    subroutine solve_triangular(zz)
      real(kind=dp), intent(inout) :: zz(:)

      do l = 1, nsetp
        ip = nsetp + 1 - l
        if (l /= 1) then
          zz(1:ip) = zz(1:ip) - a(1:ip,jj) * zz(ip + 1)
        end if
        jj = indx(ip)
        zz(ip) = zz(ip) / a(ip,jj)
      end do
    end subroutine solve_triangular

  end subroutine nnls__dp

  !> @brief Computes the orthogonal rotation matrix.
  !>
  !> @authors The original version of this code was developed by
  !> Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
  !> 1973 JUN 12, and published in the book
  !> "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
  !> Revised FEB 1995 to accompany reprinting of the book by SIAM.
  pure subroutine g1__dp(a, b, cterm, sterm, sig)
    use base_mod, only: dp

    real(kind=dp), intent(out) :: cterm, sterm, sig
    real(kind=dp), intent(in)  :: a, b
    real(kind=dp)              :: xr, yr
    real(kind=dp), parameter   :: one = 1.0_dp, zero = 0.0_dp

    if (abs( a ) > abs( b )) then
      xr = b / a
      yr = sqrt( one + xr**2 )
      cterm = sign( one / yr, a )
      sterm = cterm * xr
      sig = abs( a ) * yr
    else if (b /= zero) then
      xr = a / b
      yr = sqrt( one + xr**2 )
      sterm = sign( one / yr, b )
      cterm = sterm * xr
      sig = abs( b ) * yr
    else
      cterm = zero
      sterm = one
    end if
  end subroutine g1__dp

  !> @brief construction or application of a single Householder transformation Q = I + U * (U**T) / B
  !>
  !> @authors The original version of this code was developed by
  !> Charles L. Lawson and Richard J. Hanson at Jet Propulsion Laboratory
  !> 1973 JUN 12, and published in the book
  !> "SOLVING LEAST SQUARES PROBLEMS", Prentice-HalL, 1974.
  !> Revised FEB 1995 to accompany reprinting of the book by SIAM.
  !>
  !> @param[in] mode The modes 1 or 2 select algorithm H1 to construct and apply a
  !>        Householder transformation, or algorithm H2 to apply a
  !>        previously constructed transformation, respectively.
  !> @param[in] lpivot is the index of the pivot element.
  !> @param[in] l1 If \c l1 is less than or equal to \c m the transformation will be constructed to
  !>        zero elements indexed from \c l1 through \c m. If \c l1 is greater than \c m
  !>        the subroutine does an identity transformation.
  !> @param[in] m If \c l1 is less than or equal \c m the transformation will be constructed to
  !>        zero elements indexed from \c l1 through \c m. If \c l1 is greater than \c m
  !>        the subroutine does an identity transformation.
  !> @param[inout] u On entry with \c mode equal to 1, \c u contains the pivot
  !>        vector. On exit when \c mode equal to 1, \c u and \c up contain quantities
  !>        defining the vector u of the Householder transformation.
  !>        On entry with \c mode equal to 2, \c u and \c up should contain
  !>        quantities previously computed with \c mode equal to 1. These will
  !>        not be modified during the entry with \c mode equal to 2.
  !> @param[inout] up On exit when \c mode is equal to 1, \c u and \c up contain quantities
  !>        defining the vector u of the Householder transformation.
  !>        On entry with \c mode equal to 2, \c u and \c up should contain
  !>        quantities previously computed with \c mode equal to 1. These will
  !>        not be modified during the entry with \c mode equal to 2.
  !> @param[inout] c On entry with \c mode \c equal to 1 or 2, \c c contains a matrix,
  !>        which will be regarded as a set of vectors to which the
  !>        Householder transformation is to be applied.
  !>        on exit \c c contains the set of transformed vectors.
  !> @param[in] ice The storage increment between elements of vectors in \c c.
  !> @param[in] icv The storage increment between vectors in \c c.
  !> @param[in] ncv The number of vectors in \c c to be transformed. If \c ncv is less
  !>        than or equal to 0 no operations will be done on \c c.
  pure subroutine h12__dp(mode, lpivot, l1, m, u, up, c, ice, icv, ncv)
    use base_mod, only: dp

    integer,       intent(in)    :: mode, lpivot, l1, m, ice, icv, ncv
    real(kind=dp), intent(inout) :: u(:), c(:)
    real(kind=dp), intent(inout) :: up

    integer                      :: i, i2, i3, i4, incr, j
    real(kind=dp)                :: b, cl, clinv, sm
    real(kind=dp), parameter     :: one = 1.0_dp

    if (lpivot <= 0 .or. lpivot >= l1 .or. l1 > m) then
      ! return
    else
      cl = abs( u(lpivot) )
      if (mode /= 2) then
        do j = l1, m
          cl = max( abs( u(j) ), cl )
        end do
        if (cl <= 0) return
        clinv = one / cl
        sm = (u(lpivot) * clinv)**2 + sum( (u(l1:m) * clinv)**2 )
        cl = cl * sqrt( sm )
        if (u(lpivot) > 0) then
          cl = -cl
        end if
        up = u(lpivot) - cl
        u(lpivot) = cl
      else if (cl <= 0) then
        ! return
      else if (ncv <= 0) then
        ! return
      else
        b = up * u(lpivot)
        if (b < 0) then
          b = one / b
          i2 = 1 - icv + ice * (lpivot-1)
          incr = ice * (l1 - lpivot)
          do j = 1, ncv
            i2 = i2 + icv
            i3 = i2 + incr
            i4 = i3
            sm = c(i2) * up
            do i = l1, m
              sm = sm + c(i3) * u(i)
              i3 = i3 + ice
            end do
            if (sm /= 0) then
              sm = sm * b
              c(i2) = c(i2) + sm * up
              do i = l1, m
                c(i4) = c(i4) + sm * u(i)
                i4 = i4 + ice
              end do
            end if
          end do
        end if
      end if
    end if
  end subroutine h12__dp

end module nnls_mod

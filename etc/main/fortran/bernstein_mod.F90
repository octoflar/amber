!> @file bernstein_mod.F90
!> @author Ralf Quast
!> @date 2021
!> @copyright MIT License
!> @brief Module for evaluating multivariate Bernstein polynomials.
!>
!> Esmeralda Mainar, J.M. Pena (2006). "Evaluation algorithms for multivariate polynomials in Bernstein–Bezier form."
!> Journal of Approximation Theory 143, 44–61. <https://doi.org/10.1016/j.jat.2006.05.007>.
module bernstein_mod
  use base_mod, only: sp, dp
  implicit none
  private

  !> @brief Evaluates an univariate Bernstein basis at many abscissa values.
  interface bernstein_eval_basis
    module procedure bernstein_eval_basis__dp
  end interface

  !> @brief Evaluates an n-variate Bernstein basis at many points.
  interface bernstein_eval_basis_n
    module procedure bernstein_eval_basis_n__dp
    module procedure bernstein_eval_basis_n__sp
  end interface

  !> @brief Evaluates an univariate Bernstein polynomial at many abscissa values.
  interface bernstein_eval_poly
    module procedure bernstein_eval_poly__dp
  end interface

  !> @brief Evaluates an n-variate Bernstein polynomial at many points.
  interface bernstein_eval_poly_n
    module procedure bernstein_eval_poly_n__dp
  end interface

  !> @brief Increments the degree of an univariate Bernstein polynomial.
  interface bernstein_incr_degree
    module procedure bernstein_incr_degree__dp
  end interface

  !> @brief Increments the degree of an n-variate Bernstein polynomial.
  interface bernstein_incr_degree_n
    module procedure bernstein_incr_degree_n__dp
  end interface

  !> @brief Returns the value of an univariate Bernstein polynomial at a single abscissa value.
  interface bernstein_poly
    module procedure bernstein_poly__dp
  end interface

  !> @brief Returns the value of an n-variate Bernstein polynomial at a single point.
  interface bernstein_poly_n
    module procedure bernstein_poly_n__dp
  end interface

  public bernstein_eval_basis
  public bernstein_eval_basis_n
  public bernstein_eval_poly
  public bernstein_eval_poly_n
  public bernstein_incr_degree
  public bernstein_incr_degree_n
  public bernstein_poly
  public bernstein_poly_n

contains

  !> @brief Evaluates an univariate Bernstein basis at many abscissa values.
  !> @param[in] m The number of abscissa values.
  !> @param[in] x The abscissa values in the unit interval [0,1].
  !> @param[in] d The degree of the Bernstein basis polynomials.
  !> @param[out] y The evaluated Bernstein basis polynomials.
  !> @details The arguments are not checked.
  pure subroutine bernstein_eval_basis__dp( m, x, d, y )
    implicit none
    integer,       intent(in)  :: m
    real(kind=dp), intent(in)  :: x(m)
    integer,       intent(in)  :: d
    real(kind=dp), intent(out) :: y(0:d,m)
    integer                    :: i

    do concurrent (i = 1:m)
      call eval_de_casteljau_basis__dp( x(i), d, y(:,i) )
    end do
  end subroutine bernstein_eval_basis__dp

  !> @brief Evaluates an n-variate Bernstein basis at many points.
  !> @param[in] n The dimension.
  !> @param[in] m The number of points.
  !> @param[in] x The points in the unit n-cube.
  !> @param[in] d The degrees of the Bernstein basis polynomials.
  !> @param[out] y The evaluated Bernstein basis polynomials.
  !> @details The arguments are not checked.
  pure subroutine bernstein_eval_basis_n__dp( n, m, x, d, y )
    implicit none
    integer,       intent(in)  :: n
    integer,       intent(in)  :: m
    real(kind=dp), intent(in)  :: x(n,m)
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(out) :: y(product( d + 1 ),m)
    real(kind=dp)              :: b(product( d + 1))
    integer                    :: i, j, k
    integer                    :: l(n)

    l(1) = 1
    do i = 1, n - 1
      l(i + 1) = l(i) * (d(i) + 1)
    end do
  !TODO F18: do concurrent (k = 1:m) local(b)
    do k = 1, m
      do j = 1, product( d + 1 )
        b = 0.0_dp
        b(j) = 1.0_dp
        do i = n, 1, -1
          call eval_de_casteljau_n__dp( x(i,k), d(i), l(i), b )
        end do
        y(j,k) = b(1)
      end do
    end do
  end subroutine bernstein_eval_basis_n__dp

  pure subroutine bernstein_eval_basis_n__sp( n, m, x, d, y )
    implicit none
    integer,       intent(in)  :: n
    integer,       intent(in)  :: m
    real(kind=sp), intent(in)  :: x(n,m)
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(out) :: y(product( d + 1 ),m)
    real(kind=dp)              :: b(product( d + 1))
    integer                    :: i, j, k
    integer                    :: l(n)

    l(1) = 1
    do i = 1, n - 1
      l(i + 1) = l(i) * (d(i) + 1)
    end do
  !TODO F18: do concurrent (k = 1:m) local(b)
    do k = 1, m
      do j = 1, product( d + 1 )
        b = 0.0_dp
        b(j) = 1.0_dp
        do i = n, 1, -1
          call eval_de_casteljau_n__sp( x(i,k), d(i), l(i), b )
        end do
        y(j,k) = b(1)
      end do
    end do
  end subroutine bernstein_eval_basis_n__sp

  !> @brief Evaluates an univariate Bernstein polynomial at many abscissa values.
  !> @param[in] m The number of abscissa values.
  !> @param[in] x The abscissa values in the unit interval [0,1].
  !> @param[in] d The degree of the Bernstein polynomial.
  !> @param[in] c The Bernstein coefficients.
  !> @param[out] y The evaluated Bernstein polynomial.
  !> @details The arguments are not checked.
  pure subroutine bernstein_eval_poly__dp( m, x, d, c, y )
    implicit none
    integer,       intent(in)  :: m
    real(kind=dp), intent(in)  :: x(m)
    integer,       intent(in)  :: d
    real(kind=dp), intent(in)  :: c(0:d)
    real(kind=dp), intent(out) :: y(m)
    real(kind=dp)              :: b(m,0:d)
    integer                    :: i

    forall (i = 0:d)
      b(:,i) = c(i)
    end forall
    call eval_de_casteljau_many__dp( m, x, d, b)
    y = b(:,0)
  end subroutine bernstein_eval_poly__dp

  !> @brief Evaluates an n-variate Bernstein polynomial at many points.
  !> @param[in] n The dimension.
  !> @param[in] m The number of points in the unit n-cube.
  !> @param[in] x The points.
  !> @param[in] d The degrees of the Bernstein polynomial.
  !> @param[in] c The Bernstein coefficients.
  !> @param[out] y The evaluated Bernstein polynomial.
  !> @details The arguments are not checked.
  pure subroutine bernstein_eval_poly_n__dp( n, m, x, d, c, y )
    implicit none
    integer,       intent(in)  :: n
    integer,       intent(in)  :: m
    real(kind=dp), intent(in)  :: x(n,m)
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(in)  :: c(product( d + 1 ))
    real(kind=dp), intent(out) :: y(m)
    real(kind=dp)              :: b(product( d + 1 ))
    integer                    :: i, j
    integer                    :: l(n)

    l(1) = 1
    do i = 1, n - 1
      l(i + 1) = l(i) * (d(i) + 1)
    end do
  !TODO F18: do concurrent (j = 1:m) local(b)
    do j = 1, m
      b = c
      do i = n, 1, -1
        call eval_de_casteljau_n__dp( x(i, j), d(i), l(i), b )
      end do
      y(j) = b(1)
    end do
  end subroutine bernstein_eval_poly_n__dp

  !> @brief Increments the degree of an univariate Bernstein polynomial.
  !> @param[in] incr The increment.
  !> @param[in] d The original degree of the Bernstein polynomial.
  !> @param[in] b The original Bernstein coefficients.
  !> @param[out] c The new Bernstein coefficients of the equivalent higher-degree polynomial.
  !> @details The arguments are not checked.
  pure subroutine bernstein_incr_degree__dp( incr, d, b, c )
    implicit none
    integer,       intent(in)  :: incr
    integer,       intent(in)  :: d
    real(kind=dp), intent(in)  :: b(0:d)
    real(kind=dp), intent(out) :: c(0:d + incr)
    real(kind=dp)              :: h(0:d + incr)
    integer                    :: i, j

    c(0:d) = b
    do j = 0, incr - 1
      h(0:d + j) = c(0:d + j)
      ! corresponds to: c(j) = (b(j - 1) * j + b(j) * (d + 1 - j)) / (d + 1)
      forall (i = 1:d + j)
        c(i) = (h(i - 1) * i + h(i) * (d + j + 1 - i)) / (d + j + 1)
      end forall
      ! corresponds to: c(d + 1) = b(d)
      c(d + 1 + j) = h(d + j)
    end do
  end subroutine bernstein_incr_degree__dp

  !> @brief Increments the degree of an n-variate Bernstein polynomial.
  !> @param[in] n The dimension.
  !> @param[in] incr The increments.
  !> @param[in] d The original degree of the Bernstein polynomial.
  !> @param[in] b The original Bernstein coefficients.
  !> @param[out] c The new Bernstein coefficients of the equivalent higher-degree polynomial.
  !> @details The arguments are not checked.
  pure subroutine bernstein_incr_degree_n__dp( n, incr, d, b, c )
    implicit none
    integer,       intent(in)  :: n
    integer,       intent(in)  :: incr(n)
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(in)  :: b(product( d + 1 ))
    real(kind=dp), intent(out) :: c(product( d + 1 + incr ))
    real(kind=dp)              :: h(product( d + 1 + incr ))
    integer                    :: i, j
    integer                    :: q(n)

    c(1:product( d + 1 )) = b
    q = d
    do i = 1, n
      do j = 1, incr(i)
        h = c
        call bernstein_incr_degree_n_i__dp( n, i, q, h, c ) ! increments the degree of the i-th dimension by one
        q(i) = q(i) + 1
      end do
    end do
  end subroutine bernstein_incr_degree_n__dp

  pure subroutine bernstein_incr_degree_n_i__dp( n, i, d, b, c )
    implicit none
    integer,       intent(in)  :: n
    integer,       intent(in)  :: i
    integer,       intent(in)  :: d(n)
    real(kind=dp), intent(in)  :: b(:)
    real(kind=dp), intent(out) :: c(:)
    integer                    :: j
    integer                    :: jb, jc
    integer                    :: k
    integer                    :: l
    integer                    :: lb, lc
    integer                    :: m

    l = product( d(1:i - 1) + 1 )
    m = product( d(i + 1:n) + 1 )

    lb = l * (d(i) + 1)
    lc = l * (d(i) + 2)

    ! corresponds to: c(0) = b(0)
    jb = 0
    jc = 0
    do k = 1, m
      c(jc + 1:jc + l) = b(jb + 1:jb + l)
      jb = jb + lb
      jc = jc + lc
    end do
    ! corresponds to: c(j) = (b(j - 1) * j + b(j) * (d + 1 - j)) / (d + 1)
    do j = 1, d(i)
      jb = j * l
      jc = j * l
      do k = 1, m
        c(jc + 1:jc + l) = (b(jb + 1 - l:jb) * j + b(jb + 1:jb + l) * (d(i) + 1 - j)) / (d(i) + 1)
        jb = jb + lb
        jc = jc + lc
      end do
    end do
    ! corresponds to: c(d + 1) = b(d)
    jb = d(i) * l
    jc = d(i) * l + l
    do k = 1, m
      c(jc + 1:jc + l) = b(jb + 1:jb + l)
      jb = jb + lb
      jc = jc + lc
    end do
  end subroutine bernstein_incr_degree_n_i__dp

  !> @brief Returns the value of an univariate Bernstein polynomial at a single abscissa value.
  !> @param[in] x The abscissa value in the unit interval [0,1].
  !> @param[in] d The degree of the Bernstein polynomial.
  !> @param[in] c The Bernstein coefficients.
  !> @return The value of the Bernstein polynomial.
  !> @details The arguments are not checked.
  !> @details The arguments are not checked.
  pure function bernstein_poly__dp( x, d, c ) result(y)
    implicit none
    real(kind=dp), intent(in) :: x
    integer,       intent(in) :: d
    real(kind=dp), intent(in) :: c(0:d)
    real(kind=dp)             :: b(0:d)
    real(kind=dp)             :: y

    b = c
    call eval_de_casteljau__dp(x, d, b)
    y = b(0)
  end function bernstein_poly__dp

  !> @brief Returns the value of an n-variate Bernstein polynomial at a single point.
  !> @param[in] n The dimension.
  !> @param[in] x The point within the unit n-cube.
  !> @param[in] d The degrees of the n-variate Bernstein polynomial.
  !> @param[in] c The Bernstein coefficients.
  !> @return The value of the Bernstein polynomial.
  !> @details The arguments are not checked.
  pure function bernstein_poly_n__dp( n, x, d, c ) result(y)
    implicit none
    integer,       intent(in) :: n
    real(kind=dp), intent(in) :: x(n)
    integer,       intent(in) :: d(n)
    real(kind=dp), intent(in) :: c(product( d + 1 ))
    real(kind=dp)             :: b(product( d + 1 ))
    integer                   :: i
    integer                   :: l(n)
    real(kind=dp)             :: y

    l(1) = 1
    do i = 1, n - 1
      l(i + 1) = l(i) * (d(i) + 1)
    end do
    b = c
    do i = n, 1, -1
      call eval_de_casteljau_n__dp( x(i), d(i), l(i), b )
    end do
    y = b(1)
  end function bernstein_poly_n__dp

  !> @brief Evaluates the de Casteljau algorithm at a single abscissa value.
  !> @param[in] x The abscissa value in the unit interval [0,1].
  !> @param[in] d The degree of the Bernstein polynomial.
  !> @param[inout] b The Bernstein coefficients.
  !> @details The arguments are not checked.
  pure subroutine eval_de_casteljau__dp( x, d, b )
    implicit none
    real(kind=dp), intent(in)    :: x
    integer,       intent(in)    :: d
    real(kind=dp), intent(inout) :: b(0:d)
    real(kind=dp)                :: p
    real(kind=dp)                :: q
    integer                      :: i, j

    p = x
    q = 1.0_dp - p
    do j = 1, d
      do i = 0, d - j
        b(i) = b(i) * q + b(i + 1) * p
      end do
    end do
  end subroutine eval_de_casteljau__dp

  !> @brief Evaluates the de Casteljau algorithm along the n-th dimension of an n-variate
  !! Bernstein batch at a single abscissa value.
  !> @param[in] x The abscissa value.
  !> @param[in] d The n-th degree of the Bernstein polynomial.
  !> @param[in] l The n-th stride in the Bernstein batch.
  !> @param[inout] b The Bernstein batch.
  !> @details The arguments are not checked.
  pure subroutine eval_de_casteljau_n__dp( x, d, l, b )
    implicit none
    real(kind=dp), intent(in)    :: x
    integer,       intent(in)    :: d
    integer,       intent(in)    :: l
    real(kind=dp), intent(inout) :: b(:)
    real(kind=dp)                :: p
    real(kind=dp)                :: q
    integer                      :: i, j, k

    p = x
    q = 1.0_dp - p
    do j = 1, d
      k = 0
      do i = 0, d - j
        b(k + 1:k + l) = b(k + 1:k + l) * q + b(k + l + 1:k + l + l) * p
        k = k + l
      end do
    end do
  end subroutine eval_de_casteljau_n__dp

  pure subroutine eval_de_casteljau_n__sp( x, d, l, b )
    implicit none
    real(kind=sp), intent(in)    :: x
    integer,       intent(in)    :: d
    integer,       intent(in)    :: l
    real(kind=dp), intent(inout) :: b(:)
    real(kind=dp)                :: p
    real(kind=dp)                :: q
    integer                      :: i, j, k

    p = x
    q = 1.0_dp - p
    do j = 1, d
      k = 0
      do i = 0, d - j
        b(k + 1:k + l) = b(k + 1:k + l) * q + b(k + l + 1:k + l + l) * p
        k = k + l
      end do
    end do
  end subroutine eval_de_casteljau_n__sp

  !> @brief Evaluates the de Casteljau algorithm at many abscissa values.
  !> @param[in] m The number of abscissa values.
  !> @param[in] x The abscissa values.
  !> @param[in] d The degree of the Bernstein polynomial.
  !> @param[inout] b The Bernstein coefficients.
  !> @details The arguments are not checked.
  pure subroutine eval_de_casteljau_many__dp( m, x, d, b )
    implicit none
    integer,       intent(in)    :: m
    real(kind=dp), intent(in)    :: x(m)
    integer,       intent(in)    :: d
    real(kind=dp), intent(inout) :: b(m,0:d)
    real(kind=dp)                :: p(m)
    real(kind=dp)                :: q(m)
    integer                      :: i, j

    p = x
    q = 1.0_dp - p
    do j = 1, d
      do i = 0, d - j
        b(:,i) = b(:,i) * q + b(:,i + 1) * p
      end do
    end do
  end subroutine eval_de_casteljau_many__dp

  !> @brief Evaluates the de Casteljau algorithm along the n-th dimension of an n-variate
  !! Bernstein batch at many abscissa values.
  !> @param[in] x The abscissa values.
  !> @param[in] d The n-th degree of the Bernstein polynomial.
  !> @param[in] l The n-th stride in the Bernstein batch.
  !> @param[inout] b The Bernstein batch.
  !> @details The arguments are not checked.
  pure subroutine eval_de_casteljau_many_n__dp( m, x, d, l, b )
    implicit none
    integer,       intent(in)    :: m
    real(kind=dp), intent(in)    :: x(m)
    integer,       intent(in)    :: d
    integer,       intent(in)    :: l
    real(kind=dp), intent(inout) :: b(:,:)
    real(kind=dp)                :: p(m)
    real(kind=dp)                :: q(m)
    integer                      :: i, j, k, n

    p = x
    q = 1.0_dp - p
    do j = 1, d
      k = 0
      do i = 0, d - j
        forall (n = k + 1:k + l)
          b(:,n) = b(:,n) * q + b(:,n + l) * p
        end forall
        k = k + l
      end do
    end do
  end subroutine eval_de_casteljau_many_n__dp

  !> @brief Evaluates the de Casteljau algorithm for all Bernstein basis polynomials at a single abscissa value.
  !> @param[in] x The abscissa value.
  !> @param[in] d The degree of the Bernstein basis polynomials.
  !> @param[out] b The evaluated Bernstein basis polynomials.
  !> @details The arguments are not checked.
  pure subroutine eval_de_casteljau_basis__dp( x, d, y )
    implicit none
    real(kind=dp), intent(in)    :: x
    integer,       intent(in)    :: d
    real(kind=dp), intent(out)   :: y(0:d)
    real(kind=dp)                :: b(0:d,0:d)
    real(kind=dp)                :: p(0:d)
    real(kind=dp)                :: q(0:d)
    integer                      :: i, j

    b = 0.0_dp
    forall (i = 0:d)
      b(i,i) = 1.0_dp
    end forall
    p = x
    q = 1.0_dp - p
    do j = 1, d
      do i = 0, d - j
        b(:,i) = b(:,i) * q + b(:,i + 1) * p
      end do
    end do
    y = b(:,0)
  end subroutine eval_de_casteljau_basis__dp

end module bernstein_mod

/* Correctly rounded cbrtl function for binary64 values.

Copyright (c) 2024 Alexei Sibidanov and Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/* References:
   [1] Note on the Veltkamp/Dekker Algorithms with Directed Roundings,
       Paul Zimmermann, https://inria.hal.science/hal-04480440, February 2024.
   [2] CR-LIBM A library of correctly rounded elementary functions in double-precision,
       Catherine Daramy-Loirat, David Defour, Florent de Dinechin, Matthieu Gallet, Nicolas Gast,
       Christoph Lauter, Jean-Michel Muller, https://ens-lyon.hal.science/ensl-01529804, 2017.
   [3] Handbook of Floating-Point Arithmetic, Jean-Michel Muller, Nicolas Brunie, Florent de Dinechin,
       Claude-Pierre Jeannerod, Mioara Joldes, Vincent Lefèvre, Guillaume Melquiond, Nathalie Revol,
       Serge Torres, 2018.
 */

#define TRACE 0x8.7e890266896c9c6p-1L

#include <stdio.h>
#include <stdint.h>

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON

// anonymous structs, see https://port70.net/~nsz/c/c11/n1570.html#6.7.2.1p19
typedef union {
  long double f;
  struct __attribute__((__packed__)) {uint64_t m; uint32_t e:16; uint32_t empty:16;};
} b96u96_u;

typedef union {double f;uint64_t u;} b64u64_u;
/* s + t <- a + b, assuming |a| >= |b| */
static inline void
fast_two_sum (long double *s, long double *t, long double a, long double b)
{
  *s = a + b;
  long double e = *s - a;
  *t = b - e;
}

/* s + t <- a + b, assuming |a| >= |b| */
static inline void
fast_two_sum_double (double *s, double *t, double a, double b)
{
  *s = a + b;
  double e = *s - a;
  *t = b - e;
}

// Veltkamp's splitting: split x into xh + xl such that
// x = xh + xl exactly
// xh fits in 32 bits and |xh| <= 2^e if 2^(e-1) <= |x| < 2^e
// xl fits in 32 bits and |xl| < 2^(e-32)
// See reference [1].
static inline void
split (long double *xh, long double *xl, long double x)
{
  static const long double C = 0x1.00000001p+32L;
  long double gamma = C * x;
  long double delta = x - gamma;
  *xh = gamma + delta;
  *xl = x - *xh;
}

/* Dekker's algorithm: rh + rl = u * v
   Reference: Algorithm Mul12 from reference [2], pages 21-22.
   See also reference [3], Veltkamp splitting (Algorithm 4.9) and
   Dekker's product (Algorithm 4.10).
   The Handbook only mentions rounding to nearest, but Veltkamp's and
   Dekker's algorithms also work for directed roundings.
   See reference [1].
*/
static inline void
a_mul (long double *rh, long double *rl, long double u, long double v)
{
  long double u1, u2, v1, v2;
  split (&u1, &u2, u);
  split (&v1, &v2, v);
  *rh = u * v;
  *rl = (((u1 * v1 - *rh) + u1 * v2) + u2 * v1) + u2 * v2;
}

// Multiply exactly a and b, such that *hi + *lo = a * b.
static inline void a_mul_double (double *hi, double *lo, double a, double b) {
  *hi = a * b;
  *lo = __builtin_fma (a, b, -*hi);
}

// Return in hi+lo a 128-bit approximation of (ah + al) * (bh + bl)
static inline void
d_mul (long double *hi, long double *lo, long double ah, long double al,
       long double bh, long double bl) {
  a_mul (hi, lo, ah, bh); // exact
  *lo += ah * bl;
  *lo += al * bh;
}

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
static inline void d_mul_double (double *hi, double *lo, double ah, double al,
                                 double bh, double bl) {
  double s, t;

  a_mul_double (hi, &s, ah, bh);
  t = __builtin_fma (al, bh, s);
  *lo = __builtin_fma (ah, bl, t);
}

/* Return err, and update h,l,e such that (h+l)*2^exp is an an approximation of x^(1/3)
   with absolute error less than err. */
static double
fast_path (long double *h, long double *l, int *exp, long double x)
{
  b96u96_u v = {.f = x};
  int s = v.e >> 15; // sign bit
  int e = v.e & 0x7fff; // 0 <= e < 32767
  uint64_t m = v.m;
  if (e == 0) // subnormal
  {
    int k = __builtin_clzl (m);
    v.m = m << k;
    e -= k - 1;
  }
  // now x = (m/2^63)*2^(e-16383) with 2^63 <= m < 2^64
  v.e = 16383; // reduce v.f in [1,2)
  int i = (e + 63) % 3; // we add 63 since e can be negative
  *exp = ((e + 63) / 3) - 5482;
  // cbrt(x) = (-1)^s * cbrt(m/2^63) * 2^e * 2^(i/3)
  double xh = v.f;
  double xl = v.f - (long double) xh;

  /* the polynomial c0+c1*x+...+c5*x^5 approximates x^(1/3) on [1,2] with absolute
     error bounded by 2^-19.473 (cf cbrt.sollya) */
  static const double c[] = {0x1.e53b7c444f1cep-2, 0x1.ac2d3134803e2p-1,
                             -0x1.ddcd3b46e2071p-2, 0x1.9b95b5c19bd0bp-3,
                             -0x1.97bd99b63f65ep-5, 0x1.592445ed9c63ap-8};
  double xx = xh * xh;
  double r = 1.0 / xh;
  double x4 = __builtin_fma (c[5], xh, c[4]);
  double x2 = __builtin_fma (c[3], xh, c[2]);
  double x0 = __builtin_fma (c[1], xh, c[0]);
  x2 = __builtin_fma (x4, xx, x2);
  x0 = __builtin_fma (x2, xx, x0);
  // x0 approximates cbrt(xh) with absolute error < 2^-19.473
  double h0 = __builtin_fma (x0 * x0, x0, -xh) * r;
  /* Note: all ulp() below are for a precision of 53 bits (binary64).
     Write a = x, and x0 = a^(1/3) + e0, with |e0| < 2^-19.473.
     Then x0^3 - a = 3*a^(2/3)*e0 + 3*a^(1/3)*e0^2 + e0^3.
     Ignoring rounding errors, we have:
     h0 = (x0^3-a)/a = 3*a^(-1/3)*e0 + f with |f| = |(3*a^(1/3)*e0^2 + e0^3)/a|.
     The maximum of (3*a^(1/3)*e0^2 + e0^3)/a is attained at a=1, thus |f| < 2^-37.36.
     Since |h0| < 2^-17, ulp(h0) <= 2^-70, the difference between |(3*a^(1/3)*e0^2 + e0^3)/a|
     and 2^-37.36 is more than 4e6 ulp(h0), thus the bound 2^-37.36 clearly includes
     the rounding errors. */

#define MINUS_ONE_THIRD -0x1.5555555555555p-2
  double x1 = __builtin_fma (x0 * h0, MINUS_ONE_THIRD, x0);
  /* x0*h0/3 = (a^(1/3) + e0) * (a^(-1/3)*e0 + f/3)
             = e0 + a^(1/3)*f/3 + a^(-1/3)*e0^2 + e0*f/3
     Ignoring rounding errors, the maximum of a^(1/3)*f/3 + a^(-1/3)*e0^2 + e0*f/3
     is attained at a=2, thus:
     x0*h0/3 = e0 - e1 with |e1| < 2^-37.90.
     Thus x1 = a^(1/3) + e1 with |e1| < 2^-37.90.
     Since |x0*h0/3| < 2^-19, ulp(x0*h0/3) <= 2^-72, the difference between
     |a^(1/3)*f/3 + a^(-1/3)*e0^2 + e0*f/3| and 2^-37.90 is more than 9e7 ulp(x0*h0/3),
     thus the bound 2^-37.90 clearly includes the rounding errors. */

  double th, tl;
  a_mul_double (&th, &tl, x1, x1); // x1^2 = th + tl
  double h1 = __builtin_fma (th, x1, -xh);
  double h1l = __builtin_fma (tl, x1, -xl);
  h1 = (h1 + h1l) * r;
  /* Since x1 = a^(1/3) + e1 with |e1| < 2^-37.90, |h1| < 2^-35.64
     (this bound is attained in a=2, for x1 = a^(1/3) + 2^-37.90).
     Since h1 is a correction term, we can compute it in double precision only.
     Ignoring rounding errors again, we have:
     h1 = (x1^3-a)/a = 3*a^(-1/3)*e1 + f' with |f'| = |(3*a^(1/3)*e1^2 + e1^3)/a|.
     The maximum of (3*a^(1/3)*e1^2 + e1^3)/a is attained at a=1, thus |f'| < 2^-74.21.
     Now let us analyze rounding errors:
     * a_mul_double is exact, thus there is no rounding error
     * the rounding error in h1 is bounded by ulp(h1).
       Since |h1| < |3*a^(-1/3)*e1 + f'| < 2^-36, ulp(h1) <= 2^-89.
       This error is multiplied by r with |r| <= 1 thus contributes to at most 2^-89 in h1.
     * the rounding error in h1l is bounded by ulp(h1l).
       We have |tl| <= ulp(th), where |th| <= x1^2 < 2, thus |tl| <= ulp(th) <= 2^-52.
       Since |x1| < 2, this yields |tl*x1| < 2^-51.
       Now |xl| <= ulp(xh) <= 2^-52, thus |h1l| < 2^-51+2^-52, and the rounding error on tl
       is bounded by ulp(2^-51+2^-52) = 2^-103.
       This error is multiplied by r with |r| <= 1 thus contributes to at most 2^-103 in h1.
     * |h1 + h1l| < |3*a^(-1/3)*e1 + f' + 2^-51+2^-52| < 2^-36, thus the rounding error on
       h1 + h1l is bounded by ulp(2^-37) = 2^-89, and is multiplied by r with |r| <= 1
       thus contributes to at most 2^-89 in h1.
     * |(h1 + h1l) * r| < 2^-36, thus the rounding error in the product is bounded by
       ulp(2^-37) = 2^-89.
     The contribution of the rounding errors is thus bounded by:
     2^-89 + 2^-103 + 2^-89 + 2^-89 < 2^-87.41. This is much smaller than
     the difference between (3*a^(1/3)*e1^2 + e1^3)/a and the bound 2^-74.21,
     thus this bound covers also rounding errors. */

  /* x1*h1/3 = (a^(1/3) + e1) * (a^(-1/3)*e1 + f'/3)
             = e1 + a^(1/3)*f'/3 + a^(-1/3)*e1^2 + e1*f'/3
     The maximum of a^(1/3)*f'/3 + a^(-1/3)*e1^2 + e1*f'/3 is attained at a=2, thus:
     x1*h1/3 = e1 - e2 with |e2| < 2^-74.75.
     Thus x1 - x1*h1/3 = a^(1/3) + e2 with |e2| < 2^-74.75. */

  double corr = (x1 * h1) * MINUS_ONE_THIRD;
  /* Here we have two rounding errors:
     (a) the rounding error on x1 * h1. Since |x1| < 2 and |h1| < 2^-36, |x1*h1| < 2^-35
         thus this rounding error is bounded by ulp(2^-36) = 2^-88. This rounding error
         is multiplied by MINUS_ONE_THIRD thus contributes to at most
         2^-88 * |MINUS_ONE_THIRD| < 2^-89.58.
     (b) the rounding error on w * MINUS_ONE_THIRD where w = RND(x1*h1). Since |w| < 2^-35
         and |MINUS_ONE_THIRD| < 1/2, we have |corr| < 2^-36, thus this rounding error is
         bounded by ulp(corr) <= 2^-89.
     The sum (a) + (b) is thus bounded by 2^-89.58 + 2^-89 < 2^-88.26.
     This yields a total error bound for x1 + corr of:
     2^-74.75 + 2^-88.26 < 2^-74.749.
   */

  /* multiply (x1,corr) by 2^(i/3): sh[i]+sl[i] is a double-double approximation of 2^(i/3) */
  static const double sh[] = {1.0, 0x1.428a2f98d728bp+0, 0x1.965fea53d6e3dp+0};
  static const double sl[] = {0.0, -0x1.ddc22548ea41ep-56, -0x1.f53e999952f09p-54};
  d_mul_double (&x1, &corr, x1, corr, sh[i], sl[i]);

  const double sgn[] = {1.0, -1.0};
  *h = x1 * sgn[s];
  *l = corr * sgn[s];
  // err[i] is a bound for 2^-74.749*2^(i/3)
  static const double err[] = {0x1.31p-75, 0x1.80p-75, 0x1.e4p-75};
  return err[i];
}

long double
cr_cbrtl (long double x)
{
  b96u96_u v = {.f = x};
  int e = v.e & 0x7fff;

  // check NaN, Inf, 0: cbrtl(x) = x
  if (__builtin_expect (e == 32767 || x == 0, 0))
    return x;

  long double h, l;
  long double err = fast_path (&h, &l, &e, x);
  // if (x == TRACE) printf ("h=%La l=%La\n", h, l);
  long double left = h + (l - err);
  long double right = h + (l + err);
  if (left == right)
  {
    // multiply left by 2^e
    b96u96_u v = {.f = left};
    v.e += e;
    return v.f;
  }

  return 0;
}
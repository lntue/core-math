/* Correctly rounded hypotl function for binary80 values.

Copyright (c) 2025 Paul Zimmermann

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

/* This code assumes "long double" corresponds to the 80-bit double extended
   format.
*/

#include <stdint.h>
#include <fenv.h>

#ifdef __x86_64__
#include <x86intrin.h>
#endif 

#if (defined(__clang__) && __clang_major__ >= 14) || (defined(__GNUC__) && __GNUC__ >= 14 && __BITINT_MAXWIDTH__ && __BITINT_MAXWIDTH__ >= 128)
typedef unsigned _BitInt(128) u128;
#else
typedef unsigned __int128 u128;
#endif

// This code emulates the _mm_getcsr SSE intrinsic by reading the FPCR register.
// fegetexceptflag accesses the FPSR register, which seems to be much slower
// than accessing FPCR, so it should be avoided if possible.
// Adapted from sse2neon: https://github.com/DLTcollab/sse2neon
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
#include <arm64intr.h>
#endif

typedef struct
{
  uint16_t res0;
  uint8_t  res1  : 6;
  uint8_t  bit22 : 1;
  uint8_t  bit23 : 1;
  uint8_t  bit24 : 1;
  uint8_t  res2  : 7;
  uint32_t res3;
} fpcr_bitfield;

inline static unsigned int _mm_getcsr()
{
  union
  {
    fpcr_bitfield field;
    uint64_t value;
  } r;

#if defined(_MSC_VER) && !defined(__clang__)
  r.value = _ReadStatusReg(ARM64_FPCR);
#else
  __asm__ __volatile__("mrs %0, FPCR" : "=r"(r.value));
#endif
  static const unsigned int lut[2][2] = {{0x0000, 0x2000}, {0x4000, 0x6000}};
  return lut[r.field.bit22][r.field.bit23];
}
#endif  // defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON

typedef union {long double f; struct {uint64_t m; uint16_t e;};} b80u80_t;
typedef union {
	double f;
	struct __attribute__((packed)) {uint64_t m:52;uint32_t e:11;uint32_t s:1;};
	uint64_t u;
} b64u64_t;

static inline fexcept_t get_flags (void)
{
#if defined(__x86_64__) || defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  return _mm_getcsr ();
#else
  fexcept_t flag;
  fegetexceptflag (&flag, FE_ALL_EXCEPT);
  return flag;
#endif
}

static inline void set_flags (fexcept_t flag)
{
#if defined(__x86_64__)
  _mm_setcsr (flag);
#else
  fesetexceptflag (&flag, FE_ALL_EXCEPT);
#endif
}

// return non-zero iff x is a NaN (assuming x_exp = 0x4000)
inline static int
is_nan (b80u80_t s)
{
  int e = s.e;
  uint64_t m = s.m;
  return (e & 0x7fff) == 0x7fff && (m << 1) != 0;
}

// return non-zero iff x is a sNaN (assuming x_exp = 0x4000)
inline static int
is_snan (b80u80_t s)
{
  int e = s.e;
  uint64_t m = s.m;
  m = m << 1; // discard bit 63
  return (e & 0x7fff) == 0x7fff && m != 0 && (m >> 63) == 0;
}

/* The algorithm is as follows:
   - first swap x and y if |x| < |y| (where NaN > Inf > normal number)
   - then deal with x or y being NaN or Inf
   - if x or y is subnormal, normalize their significands mx and my
     so that 2^63 <= mx, my < 2^64
   - compute the exponent difference d = x_exp - y_exp
   - if d >= 32, we deduce directly the correct rounding
   - otherwise we compute two 128-bit integers hh and ll
     such that hh*2^128 + ll = mx^2 + (my/d)^2
   - deal with overflow
   - compute a double-double approximation sh + sl of sqrt(hh) by first
     computing a double approximation, and refining by Newton iteration
   - round sh + sl to an integer approximation th
   - in the subnormal case, shift right hh, ll and th so that the last
     significant bit of th corresponds to 2^-16445 (smallest subnormal)
   - compute the remainder r = hh - th^2, and adjust it and th so that
     0 <= r < 2th+1, thus th = floor(sqrt(hh))
   - if r=0 and ll=0, we are in the exact case, restore the inexact flag
   - return th or th+1 (with appropriate exponent) according to the rounding
     mode
*/
long double
cr_hypotl (long double x, long double y)
{
  volatile fexcept_t flag = get_flags ();
  b80u80_t sx = {.f = x}, sy = {.f = y};

  int x_exp = (sx.e & 0x7fff) - 0x3fff;
  int y_exp = (sy.e & 0x7fff) - 0x3fff;

  if (x_exp < y_exp || (x_exp == y_exp && sx.m < sy.m)) // swap x and y
  {
    sx.f = y;
    sy.f = x;
    int t = x_exp; x_exp = y_exp; y_exp = t;
    long double z = x; x = y; y = z;
  }

  // now x_exp > y_exp or (x_exp == y_exp and sx.m >= sy.m)

  if (__builtin_expect (x_exp == 0x4000, 0)) {
    // x or y is NaN or Inf
    /* According to IEEE 754-2019:
       hypot(±Inf, qNaN) is +Inf
       hypot(qNaN, ±Inf) is +Inf */
    if (is_snan (sx) || is_snan (sy))
      return x + y;
    if (is_nan (sx) || is_nan (sy)) {
      if (is_nan (sx) && is_nan (sy)) // x = y = qNaN
        return x + y;
      // now one is qNaN and the other is either Inf or a normal number
      if (x_exp == 0x4000 && y_exp == 0x4000) // x=qNaN and y=Inf (or converse)
        return 1.0L / 0.0L;
      return x + y;
    }
    // now neither x nor y is NaN, at least one is Inf
    return 1.0L / 0.0L;
  }

  if (__builtin_expect (y_exp == -0x3fff, 0)) { // y is 0 or subnormal
    if (__builtin_expect (sy.m == 0, 0)) // y = 0
    {
      /* hypot(±0, ±0) is +0 */
      if (x_exp == -0x3fff && sx.m == 0)
        return +0.0L;
      else // hypot(x,0) = |x|
      {
        sx.e &= 0x7fff; // clear sign bit
        return sx.f;    // |x|
      }
    }
    // normalize y
    int k = __builtin_clzll (sy.m);
    sy.m <<= k;
    y_exp -= k - 1;
    if (x_exp == -0x3fff) // x is subnormal too
    {
      k = __builtin_clzll (sx.m); // x cannot be 0
      sx.m <<= k;
      x_exp -= k - 1;
    }
  }

  // now x = sx.m * 2^(x_exp-63)

  int d = x_exp - y_exp;
  /* assume without loss of generality x = m with 2^63 <= m < 2^64,
     thus ulp(x) = 1, then if x^2+y^2 < (m+1/2)^2,
     hypot(x,y) rounds to m for rounding to nearest.
     This simplifies to y^2 < m+1/4.
     If d > 32, this is always true since y = my/2^d with my < 2^64,
     thus y^2 < 2^128/2^(2d) <= 2^62 < m.
     If d < 32, this is always false since y = my/2^d with my >= 2^63,
     thus y^2 >= 2^126/2^62 = 2^64 > m+1/4.
  */
  if (d >= 32) { // |y| < ulp(x) thus hypot(x,y) = |x| or nextabove(|x|)
    double z = 1.0;
    if (d == 32) {
      u128 yy = (u128) sy.m * (u128) sy.m;
      uint64_t h = yy >> 64, l = yy, m = sx.m;
#define ONE_FOURTH 0x4000000000000000ull   
      /* The midpoint case l == ONE_FOURTH and m odd cannot happen,
         since if m + 1/4 = y^2 with y = k + 1/2, then y^2 = k^2 + k + 1/4
         thus m = k^2+k is always even. */
      if (h > m || (h == m && l > ONE_FOURTH))
        z = 0x1.0000000000001p+0; // z + 0x1p-53 will round upwards for RNDN
      // else y^2 < m+1/4: z + 0x1p-53 will round downwards for RNDN
    }
    sx.e &= 0x7fff; // clear sign bit
    if (z + 0x1p-53 > z) // raises inexact, but result is inexact anyway
    {
      sx.m ++;
      if (__builtin_expect (sx.m == 0, 0)) // overflow case
      {
        // in case sx.e = 0x7ffe, this will round to +Inf as wanted
        sx.e ++;
        sx.m = 0x8000000000000000ull;
      }
    }
    return sx.f;
  }

  // now 0 <= d < 32
  int dd = d + d; // 0 <= dd < 64
  u128 xx = (u128) sx.m * (u128) sx.m;
  u128 yy = (u128) sy.m * (u128) sy.m;
  u128 hh = xx + (yy >> dd);
  u128 ll = (dd > 0) ? yy << (128 - dd) : 0;
  if (hh < xx) { // overflow
    ll = (hh << 126) | (ll >> 2);
    hh = (((u128) 1) << 126) | (hh >> 2);
    x_exp ++;
  }
  // sqrt(x^2 + y^2) = sqrt (hh + ll/2^128) * 2^(x_exp-63)
  // with 2^126 <= hh < 2^128 thus sqrt(x^2 + y^2) >= 2^x_exp

  if (__builtin_expect (x_exp >= 0x3fff, 0)) { // potential overflow
#define HUGE 0x1.fffffffffffffffep+16383L
    if (x_exp >= 0x4000) // sure overflow
      return HUGE + HUGE;
    // x_exp = 0x3fff
    // overflow for RNDN iff hh + ll/2^128 > (2^64-1/2)^2
#define HT (((u128) 0xffffffffffffffffull) << 64) // 2^128-2^64
#define LT (((u128) 0x4000000000000000ull) << 64) // 2^126 (thus 1/4)
    // in the midpoint case hh + ll/2^128 = (2^64-1/2)^2, we get overflow
    // since 2^64-1 is odd
    if (hh > HT || (hh == HT && ll > 0))
      return HUGE + 0x1p+16319L; // add 1/2 ulp(HUGE)
  }

  // now sqrt(x^2 + y^2) < 2^16384*(1-2^-65)

  /* We first compute a binary64 approximation of sqrt(hh),
     that we refine by Newton iteration. */
  b64u64_t h, l;
  int high = hh >> 127;
  h.m = (hh << (2 - high)) >> (128 - 52);
  // h.m takes the upper 54-high significant bits of hh
  h.e = 1024 + 125 + high;
  h.s = 0;
  // now 2^127 <= h < 2^128
  u128 low = hh << (54 - high); // next 74+high bits
  if (__builtin_expect (low == 0, 0))
    l.f = 0.0L;
  else
  {
    int e;
    uint64_t low_h = low >> 64;
    if (low_h)
      e = __builtin_clzll (low_h);
    else
      e = 64 + __builtin_clzll ((uint64_t) low);
    // e = clz(low)
    low <<= e;
    l.m = (low << 1) >> (128 - 52);
    l.e = 1024 + 125 + high - 53 - e;
    l.s = 0;
  }
  // l.f contains the next 53 bits of hh (ignoring bits with value 0)
  b64u64_t sh, sl;
  sh.f = __builtin_sqrt (h.f);
  // compute error term h + l - sh^2
  double err = __builtin_fma (sh.f, -sh.f, h.f);
  err += l.f;
  // if sh + sl = sqrt(h+l) then sh^2 + 2*sh*sl ~ h+l
  // thus eps == err/(2*s)
  sl.f = err / (2.0 * sh.f);
  // now sh+sl is a 64-bit approximation of sqrt(hh)
  u128 th = (0x10000000000000ull + sh.m) << 11;
  // add a magic constant to sl so that ulp(sl) = 1
#define MAGIC 0x1.8p+52
  sl.f = MAGIC + sl.f;
  th += (int16_t) (sl.m & 0x3fffffffffffful);

  /* sqrt(x^2+y^2) ~ th * 2^(x_exp - 63) with 2^63 <= th < 2^64
     thus 2^x_exp <= sqrt(x^2+y^2) < 2^(x_exp + 1)
     and since the smallest normal is 2^-16382,
     we are in the subnormal case when x_exp < -16382 = -0x3ffe */
  if (__builtin_expect (x_exp < -0x3ffe, 0)) // subnormal case
  {
    int k = -0x3ffe - x_exp; // we should have k < 64
    // shift right hh and ll by 2k bits, and th by k bits
    th >>= k;
    ll = (hh << (128 - 2 * k)) | (ll >> (2 * k));
    hh = hh >> (2 * k);
    x_exp += k - 1; // -1 due to exponent shift for subnormals
  }
  
  // compute r = hh - th^2
  u128 r = hh - th * th;
  if ((r >> 127) != 0) // th too large
  {
    // th -> th-1 thus r -> r + 2*th - 1
    r += 2 * th - 1;
    th --;
  }
  // if th was too large, then r < 2th+1 now, thus using "else" is ok
  else if (r >= 2 * th + 1) // th too small
  {
    // th -> th+1, thus r -> r - 2*th - 1
    r -= 2 * th + 1;
    th ++;
  }
    
  int exact = 0;
  if (__builtin_expect (r == 0 && ll == 0, 0)) // exact case
  {
    set_flags (flag);
    exact = 1;
  }

  double eps = 0x1p-53;

  /* In the midpoint case, we have hh + ll = (th+1/2)^2 thus
     r = th and ll = 2^126. */
  u128 thres = (u128) 1 << 126;
  if (r > th || (r == th && (ll > thres || (ll == thres && (th & 1)))))
    // for RNDN we should round upward
    eps = 0x1.8p-53;

  b80u80_t res = {.m = th, .e = x_exp + 0x3fff};
  if (!exact && 1.0 + eps > 1.0)
  {
    /* 1.0 + eps rounds to nextabove(1) either for RNDU (whatever the value
       of eps) and for RNDN when eps = 0x1.8p-53 */
    res.m ++;
    if (res.m == 0) // change of binade
    {
      res.e ++;
      res.m = (uint64_t) 1 << 63;
    }
  }
  return res.f;
}

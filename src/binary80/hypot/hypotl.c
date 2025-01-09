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
#include <stdbool.h>

#ifndef CORE_MATH_FAIL_QUIET
#include <stdio.h>
#include <stdlib.h>
#endif

#ifdef __x86_64__
#include <x86intrin.h>
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

static inline int get_rounding_mode (void)
{
  /* Warning: on __aarch64__ (for example cfarm103), FE_UPWARD=0x400000
     instead of 0x800. */
#if defined(__x86_64__) || defined(__arm64__) || defined(_M_ARM64)
  const unsigned flagp = _mm_getcsr ();
  return (flagp&(3<<13))>>3;
#else
  return fegetround ();
#endif
}

/* Split a number of exponent 0 (1 <= |x| < 2)
   into a high part fitting in 33 bits and a low part fitting in 31 bits:
   1 <= |rh| <= 2 and |rl| < 2^-32 */
static inline
void split(double* rh, double* rl, long double x) {
	static long double C = 0x1.8p+31L; // ulp(C)=2^-32
	long double y = (x + C) - C;
	/* Given the relative sizes of C and x, x + C has the same binade as C.
	   Therefore, the difference is exact. Furthermore,
	   ulp(x + C) = ulp(C) = 2^-32.
	   The rounding error in x + C is therefore less than 2^-32.
	   Thus, |x - y| < 2^-32. Note that since 2^31 <= x + C < 2^32 and the
	   difference is exact, y is a multiple of ulp(x + C) = 2^-32.
	   Since |x| < 2, and the roundings are monotonous, x + C is bounded
	   by the values obtained with |x| = 2, namely 0x1.7ffffffcp+31 and
	   0x1.80000004p+31, and likely for y, namely -2 and 2.
	   Since y is a multiple of 2^-32, this ensures y = k*2^-32
	   with |k| <= 2^-33, thus y fits in 33 bits.
	   (If |y| = 2, it trivially fits.) */
	*rh = y; // This conversion is exact by the argument above.
	*rl = x - y;
	/* 
	   |x - y| < 2^-32. Note that x and y are both multiples of
	   ulp_64(1) = 2^-63; therefore x - y too. This implies that
	   x - y = l*2^-63 with |l| < 2^31, thus rl fits in 31 bits,
	   and the difference is exact. */
}

// return non-zero iff x is a NaN
inline static int
is_nan (long double x)
{
  b80u80_t v = {.f = x};
  return ((v.e&0x7fff) == 0x7fff && (v.m != (1ul << 63)));
}

// return non-zero iff x is a sNaN 
inline static int
is_snan (long double x)
{
  b80u80_t v = {.f = x};
  return is_nan (x) && (!((v.m>>62)&1));
}

long double
cr_hypotl (long double x, long double y)
{
  const b80u80_t sx = {.f = x}, sy = {.f = y};
  int x_exp = (sx.e & 0x7fff) - 16383;
  int y_exp = (sy.e & 0x7fff) - 16383;

  printf ("x=%La x_exp=%d\n", x, x_exp);
  printf ("y=%La y_exp=%d\n", y, y_exp);

  if (__builtin_expect (x_exp == 0x4000 || y_exp == 0x4000, 0)) {
    // x or y is NaN or Inf
    /* According to IEEE 754-2019:
       hypot(±Inf, qNaN) is +Inf
       hypot(qNaN, ±Inf) is +Inf */
    printf ("x.m=%lu x.e=%u\n", sx.m, sx.e);
    printf ("is_snan(x)=%d\n", is_snan (x));
    printf ("is_snan(y)=%d\n", is_snan (y));
    if (is_snan (x) || is_snan (y))
      return x + y;
    if (is_nan (x) && is_nan (y)) // necessarily x = y = qNaN
      return x + y;
    // now only one of x, y can be qNaN
    return 1.0L / 0.0L;
  }

  /* hypot(±0, ±0) is +0 */
  return x + y;
}

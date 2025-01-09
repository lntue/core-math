/* Generate special cases for hypotl testing.

Copyright (c) 2024-2025 Sélène Corbineau and Paul Zimmermann, Inria.

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <mpfr.h>
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#include <omp.h>
#endif
#include <unistd.h>
#include <math.h>
#include <assert.h>

#ifndef CORE_MATH_TESTS
#define CORE_MATH_TESTS 1000000000UL /* total number of tests */
#endif

void doloop (int, int);
extern long double cr_hypotl (long double, long double);
extern int ref_fesetround (int);
extern void ref_init (void);
extern mpfr_rnd_t rnd2[];
extern long double ref_hypotl (long double, long double);

int rnd1[] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };

int rnd;
int verbose = 0;

typedef union {long double f; struct {uint64_t m; uint16_t e;};} b80u80_t;

#define MAX_THREADS 192

static unsigned int Seed[MAX_THREADS];

static long double
get_random (int tid)
{
  b80u80_t v;
  v.m = rand_r (Seed + tid);
  v.m |= (uint64_t) rand_r (Seed + tid) << 31;
  v.m |= (uint64_t) (rand_r (Seed + tid) & 1) << 62;
  // the low 63 bits of m are random
  v.e = rand_r (Seed + tid) & 0xffff;
  // if e is not 0 nor 0x8000 (0 or subnormal), m should have its most
  // significant bit set, otherwise it should be cleared
  // cf https://en.wikipedia.org/wiki/Extended_precision
  uint64_t t = (v.e & 0x7fff) != 0;
  v.m |= t << 63;
  return v.f;
}

/* define our own is_nan function to avoid depending from math.h */
static inline int
is_nan (long double x)
{
  b80u80_t v = {.f = x};
  return ((v.e == 0x7fff || v.e == 0xffff) && (v.m != (1ul << 63)));
}

static inline int
is_equal (long double x, long double y)
{
  if (is_nan (x))
    return is_nan (y);
  if (is_nan (y))
    return is_nan (x);
	b80u80_t v = {.f = x}, w = {.f = y};
  return v.e == w.e && v.m == w.m;
}

static void
check_aux (long double x, long double y)
{
  long double z, t;
  mpfr_t X, Y, Z;
  ref_init();
  ref_fesetround(rnd);
  mpfr_init2 (X, 64);
  mpfr_init2 (Y, 64);
  mpfr_init2 (Z, 64);
  mpfr_set_ld (X, x, MPFR_RNDN);
  mpfr_set_ld (Y, y, MPFR_RNDN);
  mpfr_flags_clear (MPFR_FLAGS_INEXACT);
  t = ref_hypotl (x, y);
  mpfr_flags_t inex1 = mpfr_flags_test (MPFR_FLAGS_INEXACT);
  fesetround(rnd1[rnd]);
  feclearexcept (FE_INEXACT);
  z = cr_hypotl (x, y);
  fexcept_t inex2;
  fegetexceptflag (&inex2, FE_INEXACT);
  mpfr_clear (X);
  mpfr_clear (Y);
  mpfr_clear (Z);
  if (!is_equal (z, t))
  {
    printf("FAIL x,y=%La,%La ref=%La z=%La\n", x,y,t,z);
#ifdef DO_NOT_ABORT
    return 1;
#else
    exit(1);
#endif
  }
  if ((inex1 == 0) && (inex2 != 0))
  {
    printf ("Spurious inexact exception for x,y=%La,%La (z=%La)\n", x, y, z);
    fflush (stdout);
#ifndef DO_NOT_ABORT
    exit(1);
#endif
  }
  if ((inex1 != 0) && (inex2 == 0))
  {
    printf ("Missing inexact exception for x,y=%La,%La (z=%La)\n", x, y, z);
    fflush (stdout);
#ifndef DO_NOT_ABORT
    exit(1);
#endif
  }
  return 0;
}

static void
check (long double x, long double y)
{
  check_aux (x, y);
  check_aux (x, -y);
  check_aux (-x, y);
  check_aux (-x, -y);
  check_aux (y, x);
  check_aux (y, -x);
  check_aux (-y, x);
  check_aux (-y, -x);
}

int
main (int argc, char *argv[])
{
  while (argc >= 2)
    {
      if (strcmp (argv[1], "--rndn") == 0)
        {
          rnd = 0;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndz") == 0)
        {
          rnd = 1;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndu") == 0)
        {
          rnd = 2;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--rndd") == 0)
        {
          rnd = 3;
          argc --;
          argv ++;
        }
      else if (strcmp (argv[1], "--verbose") == 0)
        {
          verbose = 1;
          argc --;
          argv ++;
        }
      else
        {
          fprintf (stderr, "Error, unknown option %s\n", argv[1]);
          exit (1);
        }
    }

  ref_init();
  ref_fesetround(rnd);
  fesetround(rnd1[rnd]);

  printf ("Checking random values\n");

  unsigned int seed = getpid ();
  for (int i = 0; i < MAX_THREADS; i++)
    Seed[i] = seed + i;

#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel for
#endif
	for(uint64_t n = 0; n < CORE_MATH_TESTS; n++) {
		ref_init();
		ref_fesetround(rnd);
		fesetround(rnd1[rnd]);
                int tid;
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
                tid = omp_get_thread_num ();
#else
                tid = 0;
#endif
		long double x = get_random(tid), y = get_random(tid);
		check(x, y);
	}

  return 0;
}

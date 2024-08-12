#define cr_function_under_test cr_sincos
#define ref_function_under_test ref_sincos

void doit (uint64_t n);
static inline uint64_t asuint (double f);

static inline int doloop (void)
{
  uint64_t nmin = asuint (0x0p0), nmax = asuint (0x1.fffffep+127);
#if (defined(_OPENMP) && !defined(CORE_MATH_NO_OPENMP))
#pragma omp parallel for
#endif
  for (uint64_t n = nmin; n <= nmax; n++)
  {
    doit (n);
    doit (n | 0x8000000000000000);
  }
  printf ("all ok\n");
  return 0;
}

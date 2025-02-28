static inline TYPE_UNDER_TEST random_under_test (void)
{
  /* sample in [-10,10] */
  return 20 * ((TYPE_UNDER_TEST) rand() / (TYPE_UNDER_TEST) RAND_MAX) - 10;

  /* for performance test of denormal results in the binary64 format */
  /* return -0x1.205966f2b4f1p+5*((TYPE_UNDER_TEST) rand() / (TYPE_UNDER_TEST) RAND_MAX) - 0x1.6232bdd7abcd2p+9; */
}

/* Correctly rounded exp2l function for binary64 values.

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

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON

// anonymous structs, see https://port70.net/~nsz/c/c11/n1570.html#6.7.2.1p19
typedef union {long double f; struct __attribute__((__packed__)) {uint64_t m; uint32_t e:16; uint32_t empty:16;};} b96u96_u;

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
   Reference: Algorithm Mul12 from https://ens-lyon.hal.science/ensl-01529804,
   pages 21-22.
   See also Handbook of Floating-Point Arithmetic, 2nd edition, Veltkamp
   splitting (Algorithm 4.9) and Dekker's product (Algorithm 4.10).
   The Handbook only mentions rounding to nearest, but Veltkamp's and
   Dekker's algorithms also work for directed roundings.
   See "Note on the Veltkamp/Dekker Algorithms with Directed Roundings",
   Paul Zimmermann, https://inria.hal.science/hal-04480440, February 2024.
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
static inline void a_mul_double(double *hi, double *lo, double a, double b) {
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

/* Return in hi+lo a 96-bit approximation of (ah + al) * (bh + bl), assuming
   1 <= ah+al, bh+bl < 2. */
static inline void
d_mul1 (long double *hi, long double *lo, long double ah, long double al,
        long double bh, long double bl) {
  static const long double C = 0x1.8p+32l;
  long double ahh = (C + ah) - C, bhh = (C + bh) - C;
  long double ahl = ah - ahh, bhl = bh - bhh;
  *hi = ahh * bhh; // exact since ahh and bhh have at most 32 significant bits
  long double t1 = ahh * (bhl + bl);
  long double t2 = (ahl + al) * bhh;
  long double t3 = (ahl + al) * (bhl + bl);
  *lo = t1 + (t2 + t3);
}

// Same as d_mul1, but assumes ah and bh fit into 32 bits
static inline void
d_mul2 (long double *hi, long double *lo, long double ah, long double al,
        long double bh, long double bl) {
  *hi = ah * bh; // exact
  long double t1 = ah * bl, t2 = al * bh, t3 = al * bl;
  *lo = (t1 + t2) + t3;
}

// Same as d_mul1, but assumes bh fits into 32 bits
static inline void
d_mul3 (long double *hi, long double *lo, long double ah, long double al,
        long double bh, long double bl) {
  static const long double C = 0x1.8p+32l; // ulp(C) = 2^-31
  long double ahh = (C + ah) - C, ahl = ah - ahh;
  *hi = ahh * bh; // exact
  long double t1 = ahh * bl;
  long double t2 = (ahl + al) * bh;
  long double t3 = (ahl + al) * bl;
  *lo = (t1 + t3) + t2;
}

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
static inline void d_mul_double(double *hi, double *lo, double ah, double al,
                         double bh, double bl) {
  double s, t;

  a_mul_double(hi, &s, ah, bh);
  t = __builtin_fma(al, bh, s);
  *lo = __builtin_fma(ah, bl, t);
}

// T2fast[i] approximates 2^(i/2^5) with absolute error < 2^-107.22
static const double T2fast[32][2] = {
   {0x1p+0L, 0x0p+0L},
   {0x1.059b0d3158574p+0L, 0x1.d73e2a475b465p-55L},
   {0x1.0b5586cf9890fp+0L, 0x1.8a62e4adc610bp-54L},
   {0x1.11301d0125b51p+0L, -0x1.6c51039449b3ap-54L},
   {0x1.172b83c7d517bp+0L, -0x1.19041b9d78a76p-55L},
   {0x1.1d4873168b9aap+0L, 0x1.e016e00a2643cp-54L},
   {0x1.2387a6e756238p+0L, 0x1.9b07eb6c70573p-54L},
   {0x1.29e9df51fdee1p+0L, 0x1.612e8afad1255p-55L},
   {0x1.306fe0a31b715p+0L, 0x1.6f46ad23182e4p-55L},
   {0x1.371a7373aa9cbp+0L, -0x1.63aeabf42eae2p-54L},
   {0x1.3dea64c123422p+0L, 0x1.ada0911f09ebcp-55L},
   {0x1.44e086061892dp+0L, 0x1.89b7a04ef80dp-59L},
   {0x1.4bfdad5362a27p+0L, 0x1.d4397afec42e2p-56L},
   {0x1.5342b569d4f82p+0L, -0x1.07abe1db13cadp-55L},
   {0x1.5ab07dd485429p+0L, 0x1.6324c054647adp-54L},
   {0x1.6247eb03a5585p+0L, -0x1.383c17e40b497p-54L},
   {0x1.6a09e667f3bcdp+0L, -0x1.bdd3413b26456p-54L},
   {0x1.71f75e8ec5f74p+0L, -0x1.16e4786887a99p-55L},
   {0x1.7a11473eb0187p+0L, -0x1.41577ee04992fp-55L},
   {0x1.82589994cce13p+0L, -0x1.d4c1dd41532d8p-54L},
   {0x1.8ace5422aa0dbp+0L, 0x1.6e9f156864b27p-54L},
   {0x1.93737b0cdc5e5p+0L, -0x1.75fc781b57ebcp-57L},
   {0x1.9c49182a3f09p+0L, 0x1.c7c46b071f2bep-56L},
   {0x1.a5503b23e255dp+0L, -0x1.d2f6edb8d41e1p-54L},
   {0x1.ae89f995ad3adp+0L, 0x1.7a1cd345dcc81p-54L},
   {0x1.b7f76f2fb5e47p+0L, -0x1.5584f7e54ac3bp-56L},
   {0x1.c199bdd85529cp+0L, 0x1.11065895048ddp-55L},
   {0x1.cb720dcef9069p+0L, 0x1.503cbd1e949dbp-56L},
   {0x1.d5818dcfba487p+0L, 0x1.2ed02d75b3707p-55L},
   {0x1.dfc97337b9b5fp+0L, -0x1.1a5cd4f184b5cp-54L},
   {0x1.ea4afa2a490dap+0L, -0x1.e9c23179c2893p-54L},
   {0x1.f50765b6e454p+0L, 0x1.9d3e12dd8a18bp-54L},
};

// T1fast[i] approximates 2^(i/2^10) with absolute error < 2^-107.03
static const double T1fast[32][2] = {
   {0x1p+0L, 0x0p+0L},
   {0x1.002c605e2e8cfp+0L, -0x1.d7c96f201bb2fp-55L},
   {0x1.0058c86da1c0ap+0L, -0x1.5e00e62d6b30dp-56L},
   {0x1.0085382faef83p+0L, 0x1.da93f90835f75p-56L},
   {0x1.00b1afa5abcbfp+0L, -0x1.4f6b2a7609f71p-55L},
   {0x1.00de2ed0ee0f5p+0L, -0x1.406ac4e81a645p-57L},
   {0x1.010ab5b2cbd11p+0L, 0x1.c1d0660524e08p-54L},
   {0x1.0137444c9b5b5p+0L, -0x1.2b6aeb6176892p-56L},
   {0x1.0163da9fb3335p+0L, 0x1.b61299ab8cdb7p-54L},
   {0x1.019078ad6a19fp+0L, -0x1.008eff5142bf9p-56L},
   {0x1.01bd1e77170b4p+0L, 0x1.5e7626621eb5bp-56L},
   {0x1.01e9cbfe113efp+0L, -0x1.c11f5239bf535p-55L},
   {0x1.02168143b0281p+0L, -0x1.2bf310fc54eb6p-55L},
   {0x1.02433e494b755p+0L, -0x1.314aa16278aa3p-54L},
   {0x1.027003103b10ep+0L, -0x1.082ef51b61d7ep-56L},
   {0x1.029ccf99d720ap+0L, 0x1.64cbba902ca27p-58L},
   {0x1.02c9a3e778061p+0L, -0x1.19083535b085dp-56L},
   {0x1.02f67ffa765e6p+0L, -0x1.b8db0e9dbd87ep-55L},
   {0x1.032363d42b027p+0L, 0x1.fea8d61ed6016p-54L},
   {0x1.03504f75ef071p+0L, 0x1.bc2ee8e5799acp-54L},
   {0x1.037d42e11bbccp+0L, 0x1.56811eeade11ap-57L},
   {0x1.03aa3e170aafep+0L, -0x1.f1a93c1b824d3p-54L},
   {0x1.03d7411915a8ap+0L, 0x1.b7c00e7b751dap-54L},
   {0x1.04044be896ab6p+0L, 0x1.9dc3add8f9c02p-54L},
   {0x1.04315e86e7f85p+0L, -0x1.0a31c1977c96ep-54L},
   {0x1.045e78f5640b9p+0L, 0x1.35bc86af4ee9ap-56L},
   {0x1.048b9b35659d8p+0L, 0x1.21cd53d5e8b66p-57L},
   {0x1.04b8c54847a28p+0L, -0x1.e7992580447bp-56L},
   {0x1.04e5f72f654b1p+0L, 0x1.4c3793aa0d08dp-55L},
   {0x1.051330ec1a03fp+0L, 0x1.79a8be239ca45p-54L},
   {0x1.0540727fc1762p+0L, -0x1.abcae24b819dfp-54L},
   {0x1.056dbbebb786bp+0L, 0x1.06c87433776c9p-55L},
};

// T0fast[i] approximates 2^(i/2^15) with absolute error < 2^-107.21
static const double T0fast[32][2] = {
  {0x1p+0L, 0x0p+0L},
   {0x1.000162e525eep+0L, 0x1.51d5115f56655p-54L},
   {0x1.0002c5cc37da9p+0L, 0x1.247426170d232p-54L},
   {0x1.000428b535c85p+0L, 0x1.fb74d9ea60832p-54L},
   {0x1.00058ba01fbap+0L, -0x1.a4a4d4cad39fep-54L},
   {0x1.0006ee8cf5b22p+0L, 0x1.932ef86740288p-55L},
   {0x1.0008517bb7b38p+0L, -0x1.9bcb5db05e94p-57L},
   {0x1.0009b46c65c0bp+0L, 0x1.eb71a14c21e8bp-54L},
   {0x1.000b175effdc7p+0L, 0x1.ae8e38c59c72ap-54L},
   {0x1.000c7a5386096p+0L, 0x1.9efe59410befap-54L},
   {0x1.000ddd49f84a3p+0L, 0x1.1b41ae4029256p-56L},
   {0x1.000f404256a18p+0L, 0x1.87fa20970e17ap-57L},
   {0x1.0010a33ca112p+0L, -0x1.68ddbffb2ac39p-58L},
   {0x1.00120638d79e5p+0L, 0x1.fcfcbaad3ac82p-54L},
   {0x1.00136936fa493p+0L, 0x1.f2be4da91d517p-55L},
   {0x1.0014cc3709154p+0L, -0x1.257410422c2fdp-55L},
   {0x1.00162f3904052p+0L, -0x1.7b5d0d58ea8f4p-58L},
   {0x1.0017923ceb1b8p+0L, 0x1.f5e282a52dbd9p-55L},
   {0x1.0018f542be5b1p+0L, 0x1.36ad1777e482p-54L},
   {0x1.001a584a7dc68p+0L, -0x1.a447def06db7ep-55L},
   {0x1.001bbb5429606p+0L, 0x1.73c902846716ep-54L},
   {0x1.001d1e5fc12b8p+0L, -0x1.6354c4339b91p-54L},
   {0x1.001e816d452a6p+0L, 0x1.3da68462bd1e4p-54L},
   {0x1.001fe47cb55fdp+0L, -0x1.334e0c9692b31p-58L},
   {0x1.0021478e11ce6p+0L, 0x1.4115cb6b16a8ep-54L},
   {0x1.0022aaa15a78dp+0L, -0x1.6c81d3063bdb2p-57L},
   {0x1.00240db68f61cp+0L, -0x1.c65136ca57a55p-54L},
   {0x1.002570cdb08bdp+0L, -0x1.ded5dcc6c5bd4p-55L},
   {0x1.0026d3e6bdf9bp+0L, 0x1.e3a2b72b6b281p-55L},
   {0x1.00283701b7ae2p+0L, -0x1.870119822944dp-54L},
   {0x1.00299a1e9dabbp+0L, -0x1.bd5a8a6af3c4ep-54L},
   {0x1.002afd3d6ff51p+0L, -0x1.13c6aeb99597p-54L},
};

// T2[i] approximates 2^(i/2^5) with relative error < 2^-129.565
static const long double T2[32][2] = {
   {0x1p+0L, 0x0p+0L},
   {0x1.059b0d31585743aep+0L, 0x1.f1523ada32905ffap-66L},
   {0x1.0b5586cf9890f62ap+0L, -0x1.d1b5239ef559f27p-66L},
   {0x1.11301d0125b50a4ep+0L, 0x1.77e35db26319d58cp-65L},
   {0x1.172b83c7d517adcep+0L, -0x1.06e75e29d6b0dbfap-69L},
   {0x1.1d4873168b9aa78p+0L, 0x1.6e00a2643c1ea62ep-66L},
   {0x1.2387a6e75623866cp+0L, 0x1.fadb1c15cb593b04p-68L},
   {0x1.29e9df51fdee12c2p+0L, 0x1.7457d6892a8ef2a2p-66L},
   {0x1.306fe0a31b7152dep+0L, 0x1.1ab48c60b90bdbdap-65L},
   {0x1.371a7373aa9caa72p+0L, -0x1.755fa17570cf0384p-65L},
   {0x1.3dea64c12342235cp+0L, -0x1.7dbb83d8511808bap-65L},
   {0x1.44e086061892d032p+0L, -0x1.9217ec41fcc08562p-65L},
   {0x1.4bfdad5362a271d4p+0L, 0x1.cbd7f621710701b2p-67L},
   {0x1.5342b569d4f81dfp+0L, 0x1.507893b0d4c7e9ccp-65L},
   {0x1.5ab07dd48542958cp+0L, 0x1.2602a323d668bb12p-65L},
   {0x1.6247eb03a5584b2p+0L, -0x1.e0bf205a4b7a89c6p-65L},
   {0x1.6a09e667f3bcc908p+0L, 0x1.65f626cdd52afa7cp-65L},
   {0x1.71f75e8ec5f73dd2p+0L, 0x1.b879778566b65a1ap-67L},
   {0x1.7a11473eb0186d7ep+0L, -0x1.5dfb81264bc14218p-65L},
   {0x1.82589994cce128acp+0L, 0x1.f115f56694021ed6p-65L},
   {0x1.8ace5422aa0db5bap+0L, 0x1.f156864b26ecf9bcp-66L},
   {0x1.93737b0cdc5e4f46p+0L, -0x1.fc781b57ebba5a08p-65L},
   {0x1.9c49182a3f0901c8p+0L, -0x1.dca7c706a0d3912ap-67L},
   {0x1.a5503b23e255c8b4p+0L, 0x1.2248e57c3de40286p-67L},
   {0x1.ae89f995ad3ad5e8p+0L, 0x1.cd345dcc8169fefp-66L},
   {0x1.b7f76f2fb5e46eaap+0L, 0x1.ec206ad4f14d5322p-66L},
   {0x1.c199bdd85529c222p+0L, 0x1.9625412374ccf288p-69L},
   {0x1.cb720dcef906915p+0L, 0x1.e5e8f4a4edbb0ecap-67L},
   {0x1.d5818dcfba48725ep+0L, -0x1.7e9452647c8d582ap-66L},
   {0x1.dfc97337b9b5eb96p+0L, 0x1.195873da5236e44cp-65L},
   {0x1.ea4afa2a490d9858p+0L, 0x1.ee7431ebb6603f0ep-65L},
   {0x1.f50765b6e4540674p+0L, 0x1.f096ec50c575ff32p-65L},
};

// T1[i] approximates 2^(i/2^10) with relative error < 2^-129.048
static const long double T1[32][2] = {
   {0x1p+0L, 0x0p+0L},
   {0x1.002c605e2e8cec5p+0L, 0x1.b486ff22688e8042p-66L},
   {0x1.0058c86da1c09ea2p+0L, -0x1.cc5ad661a130c72ep-73L},
   {0x1.0085382faef831dap+0L, 0x1.27f2106beea70f16p-65L},
   {0x1.00b1afa5abcbed62p+0L, -0x1.aca9d827dc46d578p-65L},
   {0x1.00de2ed0ee0f4f6p+0L, -0x1.ab13a069914e78d8p-67L},
   {0x1.010ab5b2cbd11708p+0L, -0x1.7ccfd6d8fbc56654p-65L},
   {0x1.0137444c9b5b4ed4p+0L, 0x1.2a293d12edc0f6d8p-65L},
   {0x1.0163da9fb33356d8p+0L, 0x1.299ab8cdb737e9p-66L},
   {0x1.019078ad6a19efp+0L, -0x1.1dfea2857f2adcfap-65L},
   {0x1.01bd1e77170b415ep+0L, 0x1.d899887ad6abfd84p-66L},
   {0x1.01e9cbfe113eec7ep+0L, -0x1.f5239bf535594f58p-67L},
   {0x1.02168143b0280da8p+0L, 0x1.9de0756294cca9f6p-68L},
   {0x1.02433e494b754b3ap+0L, 0x1.aaf4ec3aae71c11ep-65L},
   {0x1.027003103b10def8p+0L, -0x1.77a8db0ebeced796p-67L},
   {0x1.029ccf99d720a05ap+0L, -0x1.9a22b7e9aec548fp-65L},
   {0x1.02c9a3e778060ee6p+0L, 0x1.ef95949ef4537bd2p-65L},
   {0x1.02f67ffa765e5c8ep+0L, 0x1.278b1213c0c9e1b6p-66L},
   {0x1.032363d42b0277fap+0L, 0x1.46b0f6b00b29401ep-65L},
   {0x1.03504f75ef0716fp+0L, 0x1.77472bccd623cb4ap-65L},
   {0x1.037d42e11bbcc0acp+0L, -0x1.7ee11521ee5bb3bp-65L},
   {0x1.03aa3e170aafd83ap+0L, -0x1.49e0dc1269659b0ep-65L},
   {0x1.03d7411915a8a6ep+0L, -0x1.ff8c2457133e5c34p-65L},
   {0x1.04044be896ab6678p+0L, -0x1.e2913831fef18048p-65L},
   {0x1.04315e86e7f84bd8p+0L, -0x1.8e0cbbe4b703226p-65L},
   {0x1.045e78f5640b9136p+0L, -0x1.0de542c45976151ep-66L},
   {0x1.048b9b35659d809p+0L, 0x1.cd53d5e8b6609244p-65L},
   {0x1.04b8c54847a27e18p+0L, 0x1.9b69feee140b2d6cp-66L},
   {0x1.04e5f72f654b1298p+0L, 0x1.bc9d50684640c7dap-66L},
   {0x1.051330ec1a03f5e6p+0L, 0x1.45f11ce522be682ep-65L},
   {0x1.0540727fc176195p+0L, 0x1.a8eda3f31093fe7cp-65L},
   {0x1.056dbbebb786b20ep+0L, -0x1.bc5e64449ba34522p-66L},
};

// T0[i] approximates 2^(i/2^15) with relative error < 2^-129.004
static const long double T0[32][2] = {
   {0x1p+0L, 0x0p+0L},
   {0x1.000162e525ee0548p+0L, -0x1.5775054cd5adbfb2p-65L},
   {0x1.0002c5cc37da9492p+0L, -0x1.7b3d1e5b9cb8c262p-67L},
   {0x1.000428b535c857eep+0L, -0x1.64c2b3ef9bd797e4p-67L},
   {0x1.00058ba01fb9f96ep+0L, -0x1.26a6569cfedd0784p-65L},
   {0x1.0006ee8cf5b22326p+0L, 0x1.77c33a014414bc8ep-66L},
   {0x1.0008517bb7b37f32p+0L, 0x1.a5127d0b5ff94c8cp-68L},
   {0x1.0009b46c65c0b7aep+0L, -0x1.cbd67bc2e9bcfbf6p-67L},
   {0x1.000b175effdc76bap+0L, 0x1.c718b38e549cb934p-67L},
   {0x1.000c7a538609667cp+0L, -0x1.a6bef4105b137bf2p-70L},
   {0x1.000ddd49f84a311cp+0L, -0x1.7ca37fadb538a1d8p-65L},
   {0x1.000f404256a180c4p+0L, -0x1.77da3c7a168d87dap-71L},
   {0x1.0010a33ca111ffa6p+0L, -0x1.bb7ff655871c632cp-67L},
   {0x1.00120638d79e57f4p+0L, -0x1.a2a9629bed7b0238p-69L},
   {0x1.00136936fa4933e6p+0L, -0x1.06c95b8aba5aab5ep-65L},
   {0x1.0014cc3709153db6p+0L, -0x1.d04108b0bf2a604p-65L},
   {0x1.00162f3904051fa2p+0L, -0x1.ae86ac75479c344p-65L},
   {0x1.0017923ceb1b83ecp+0L, -0x1.d7d5ad2426d98758p-67L},
   {0x1.0018f542be5b14dap+0L, 0x1.68bbbf240fe795acp-65L},
   {0x1.001a584a7dc67cb8p+0L, -0x1.1f7bc1b6df8284a4p-65L},
   {0x1.001bbb54296065dp+0L, -0x1.b7ebdcc748e85934p-65L},
   {0x1.001d1e5fc12b7a72p+0L, 0x1.59de63237804a4cep-65L},
   {0x1.001e816d452a64f6p+0L, 0x1.342315e8f1e6f0fap-65L},
   {0x1.001fe47cb55fcfb4p+0L, -0x1.a7064b4959898e28p-65L},
   {0x1.0021478e11ce6504p+0L, 0x1.5cb6b16a8e0ad03cp-66L},
   {0x1.0022aaa15a78cf4ap+0L, -0x1.03a60c77b646fde4p-66L},
   {0x1.00240db68f61b8e6p+0L, 0x1.7649ad42d581bc88p-65L},
   {0x1.002570cdb08bcc42p+0L, 0x1.5119c9d215fbae7p-66L},
   {0x1.0026d3e6bdf9b3c8p+0L, -0x1.752352535fcc167ep-65L},
   {0x1.00283701b7ae19e4p+0L, -0x1.19822944d4228146p-70L},
   {0x1.00299a1e9daba90ap+0L, 0x1.2baca861d8c8d1f4p-65L},
   {0x1.002afd3d6ff50bbp+0L, 0x1.ca8a335347ceeba2p-65L},
};

// put in h+l an approximation of 2^x for |xh+xl| < 2^-16, with relative error
// bounded by 2^-86.887 (see routine analyze_P in exp2l.sage), and |l| < 2^-51.999
// At input we have |xh|,|xh+xl| <= 2^-16 and |xl| <= 2^-69
static void
P (double *h, double *l, double xh, double xl)
{
  /* the following degree-4 polynomial generated by exp2.sollya has absolute
     error bounded by 2^-90.627 for |x| < 2^-16 */
  static const double p[] = {1.0L,                                        // degree 0
                             0x1.62e42fefa39efp-1, 0x1.abc9c864cbd56p-56, // degree 1
                             0x1.ebfbdff82c58fp-3,                        // degree 2
                             0x1.c6b08d7057b35p-5,                        // degree 3
                             0x1.3b2a52e855b32p-7};                       // degree 4
  double y = p[5] * xh + p[4];
  y = y * xh + p[3];
  a_mul_double (h, l, y, xh);
  // add p[1]+p[2]
  double t;
  fast_two_sum_double (h, &t, p[1], *h);
  *l += t + p[2];
  // multiply by xh+xl
  d_mul_double (h, l, *h, *l, xh, xl);
  // add p[0]
  fast_two_sum_double (h, &t, p[0], *h);
  *l += t;
}

// put in h+l an approximation of 2^x for |x| < 2^-16, with relative error
// bounded by 2^-125.403 (see routine analyze_Pacc in exp2l.sage)
// and |l| < 2^-62.999
static void
Pacc (long double *h, long double *l, long double x)
{
  /* the following degree-6 polynomial generated by exp2acc.sollya has absolute
     error bounded by 2^-133.987 for |x| < 2^-16 */
  static const long double p[] = {1.0L, // degree 0
                                  0x1.62e42fefa39ef358p-1L, -0x1.b0e2633fe0676a9cp-67L, // degree 1
                                  0x1.ebfbdff82c58ea86p-3L, 0x1.e2d60dd936b9ba5ep-68L,  // degree 2
                                  0x1.c6b08d704a0bf8b4p-5L, -0x1.8b4ba2fbcf44117p-70L,  // degree 3
                                  0x1.3b2ab6fba4e7729cp-7L,  // degree 4
                                  0x1.5d87fe78ad725bcep-10L, // degree 5
                                  0x1.4309131bde9fabeap-13L, // degree 6
  };
  long double y = p[9] * x + p[8]; // a6*x+a5
  y = y * x + p[7];                // y*x+a4
  y = y * x;                       // y*x
  fast_two_sum (h, l, p[5], y);    // a3h+y
  *l += p[6];                      // +a3l
  // multiply h+l by x
  long double t;
  a_mul (h, &t, *h, x);            // exact
  *l = *l * x + t;
  // add a2h+a2l
  fast_two_sum (h, &t, p[3], *h);
  *l += t + p[4];
  // multiply h+l by x
  a_mul (h, &t, *h, x);            // exact
  *l = *l * x + t;
  // add a1h+a1l
  fast_two_sum (h, &t, p[1], *h);
  *l += t + p[2];
  // multiply h+l by x
  a_mul (h, &t, *h, x);            // exact
  *l = *l * x + t;
  // add a0
  fast_two_sum (h, &t, p[0], *h);
  *l += t;
}

#define TRACE 0x8.18c728ec71a98p-19L

/* Assume -16446 < x < -0x1.71547652b82fe176p-65
   or 0x1.71547652b82fe176p-64 < x < 16384.
   Return H + L approximating 2^x with relative error < 2^-85.803
   or H = L = NaN.
*/
static void
fast_path (long double *H, long double *L, long double x)
{
  b96u96_u v = {.f = x};

  // compute k = round(2^15*x)
  int64_t s = 48 - ((v.e&0x7fff) - 0x3fff);
  // if e = (v.e&0x7fff) - 0x3fff, we have 2^e <= |x| < 2^(e+1)
  // thus with s = 48 - e: 2^(48-s) <= |x| < 2^(49-s)
  // With the input range for x we have -65 <= e <= 14 thus 34 <= s <= 113
  uint64_t m = v.m + (1l<<(s-1)), sgn = -(v.e>>15);
  // x = (-1)^sgn*m/2^63*2^e
  // bit s in v.m (when x is multiplied by 2^15) corresponds to 1, thus bit s-1 to 1/2
  if(__builtin_expect(m<v.m,0)) { // exponent shift in 2^15*x+1/2
    s--;
    m = (v.m>>1) + (1l<<(s-1));
  }
  // bit s in v.m corresponds to 1 in 2^15*x+1/2
  if(s>63) m = 0; // |x| < 2^-16
  m >>= s; // round to integer
  m = (m^sgn) - sgn; // gives m for sgn = 0, -m otherwise
  int32_t k = m; // -16445*2^15 <= k <= 16383*2^15

  long double r = x - (long double) k * 0x1p-15L;
  /* Now |r| <= 2^-16 and r is an integer multiple of ulp(x).
     If |x| >= 2^-6, then ulp(x) >= 2^-69, thus r is exactly representable as double.
  */
  double rh, rl;
  rh = r;
  rl = r - (long double) rh;
  /* Since |rh| <= 2^-16, we have |rl| <= ulp(2^-17) = 2^-69. */
  int32_t i = (k + 538869760) & 32767;
  int32_t e = (k - i) >> 15;
  int32_t i0 = i & 0x1f, i1 = (i >> 5) & 0x1f, i2 = i >> 10;
  // k = e*2^15 + i2*2^10 + i1*2^5 + i0
  // x = k*2^-15 + r with |r| < 2^-16
  // 2^x = 2^e * 2^(i2/2^5) * 2^(i1/2^10) * 2^(i0/2^15) * 2^r
  double h, l, hh, ll;
  P (&h, &l, rh, rl); // relative error bounded by 2^-86.887, with |l| < 2^-51.999
  d_mul_double (&hh, &ll, T2fast[i2][0], T2fast[i2][1], T1fast[i1][0], T1fast[i1][1]);
  /* We have |T2fast[i2][0]|, |T1fast[i1][0]| < 2, |T2fast[i2][1]|, |T1fast[i1][1]| < 2^-53.
     The call err_d_mul(2.,2^-53.,2.,2^-53.) with err_d_mul from exp2l.sage
     yields (4.06756404254584e-31, 1.33226762955019e-15) thus the absolute error is bounded
     by 2^-100.95 and |ll| < 2^-49.415.
  */
  d_mul_double (&hh, &ll, hh, ll, T0fast[i0][0], T0fast[i0][1]);
  /* We have |hh_in|, |T0fast[i0][0]| < 2, |ll_in| < 2^-49.415, |T0fast[i0][1]| < 2^-53.
     The call err_d_mul(2.,2^-49.415,2.,2^-53.) with err_d_mul from exp2l.sage
     yields (1.72563707481138e-30, 3.77482754261006e-15) thus the absolute error is bounded
     by 2^-98.87 and |ll| < 2^-47.912.
     If we add the input error on hh_in+ll_in bounded by 2^-100.95 and multiplied by
     T0fast[i0][0]+T0fast[i0][1] < 2, this yields a total absolute error of:
     2^-98.87 + 2^-100.95*2 < 2^-98.311.
  */
  d_mul_double (&h, &l, h, l, hh, ll);
  /* At input, we have 0.999989 < h_in + l_in < 1.000011, |l_in| < 2^-51.999,
     and 1 <= hh + ll < 1.999958, |ll| < 2^-47.912, thus at output 0.999989 < h + l < 1.999980.
     The call err_d_mul(1.000011,2^-51.999,1.999958,2^-47.912) with err_d_mul from exp2l.sage
     yields (1.72652675153480e-30, 4.44263794756115e-15) thus the absolute error is bounded
     by 2^-98.86 and |ll| < 2^-47.677.
     If we add the input error on hh+ll bounded by 2^-98.311 and multiplied by
     h_in+l_in < 1.000011, and the input error on h_in+l_in bounded by 2^-86.887*1.000011
     and multiplied by hh+ll < 1.999958, this yields a total absolute error of:
     2^-98.86 + 1.000011*2^-98.311 + 2^-86.887*1.000011*1.999958 < 2^-85.886.
     Since |h+l| > 0.999989 this yields a relative error of at most:
     2^-85.886/0.999989 < 2^-85.885.
   */
  if (__builtin_expect (e >= -16355, 1))
  {
    /* Multiply h, l by 2^e. Since e >= -16355, we have 2^x>=0.99998*2^-16355
       thus if l*2^e is in the subnormal range, we have an additional absolute
       error of at most 2^-16445, which corresponds to an additional relative
       error < 2^-16445/(0.99998*2^-16355) < 2^-89.999. This gives a final
       bound of (1 + 2^-85.885) * (1 + 2^-89.999) - 1 < 2^-85.803.
       No overflow is possible here since x < 16384. */
    // since |h| > 0.5, |h*2^e| > 2^-16356 and is exactly representable
    v.f = h;
    v.e += e;
    *H = v.f;
    b96u96_u w = {.f = l};
    if (__builtin_expect ((w.e & 0x7fff) + e > 0, 1))
      {
        w.e += e;
        *L = w.f;
      }
    else
      *L = __builtin_ldexpl (l, e);
  }
  else
  {
    v.e = 32767;
    v.m = 0xc000000000000000ul;
    *H = *L = v.f; // +qnan
  }
}

static void
accurate_path (long double *h, long double *l, long double x)
{
#define EXCEPTIONS 152
static const long double exceptions[EXCEPTIONS][3] = {
    {-0xb.8aa3b295c17f0bcp-68L, 0x1.fffffffffffffffep-1L, 0x1.fffffffffffffffep-66L},
    {-0xd.b4a26411d5c6de3p-64L, 0x1.ffffffffffffffeep-1L, -0x1.fffffffffffffffep-66L},
    {-0xb.738e6b3095fc0f1p-61L, 0x1.ffffffffffffff82p-1L, -0x1.fffffffffffffffep-66L},
    {-0xb.fe0e178f9b0e03cp-61L, 0x1.ffffffffffffff7ap-1L, 0x1.fffffffffffffffep-66L},
    {-0xe.2dd21ae4fa369cap-59L, 0x1.fffffffffffffd8ap-1L, 0x1.fffffffffffffffep-66L},
    {-0xc.81571d9f0287227p-57L, 0x1.fffffffffffff756p-1L, -0x1.fffffffffffffffep-66L},
    {-0xd.4759a60edbc72d3p-55L, 0x1.ffffffffffffdb2ep-1L, 0x1.fffffffffffffffep-66L},
    {-0x9.dd2c3c92e75472bp-49L, 0x1.fffffffffff929aep-1L, 0x1.fffffffffffffffep-66L},
    {-0xf.80d811a47bbbfb8p-44L, 0x1.fffffffffea8203ap-1L, -0x1.fffffffffffffffep-66L},
    {-0xe.ceb5c7152d2355cp-40L, 0x1.ffffffffeb78f5c2p-1L, 0x1.fffffffffffffffep-66L},
    {-0xd.15f3e165e7e7965p-29L, 0x1.ffffff6edfd62336p-1L, 0x1.fffffffffffffffep-66L},
    {-0x1.8acea303b9027c9ep-16L, 0x1.fffddcaf7d274244p-1L, 0x1.99115086dbbc6382p-125L},
    {-0x1.9632fa92f69b0c6p-16L, 0x1.fffdcce4a5b33a76p-1L, -0x1.fffffffffffffffep-66L},
    {-0x1.c483049c00e8420cp-16L, 0x1.fffd8cb0fa80cba6p-1L, 0x1.8ab5cb057dbb0a8ep-126L},
    {-0x1.d8a770dbc7a8d258p-16L, 0x1.fffd70c4cb93c8bap-1L, 0x1.5f6c1dcddd5cb248p-126L},
    {-0x1.dc3b0251ef38ec24p-16L, 0x1.fffd6bcf92b3de66p-1L, 0x1.fffffffffffffffcp-66L},
    {-0x1.e4510ac98ee39be4p-16L, 0x1.fffd6099f46ef5e2p-1L, 0x1.70785dc4175eaa92p-125L},
    {-0x1.30bdeead0ab9134cp-15L, 0x1.fffcb3162ca0ab18p-1L, 0x1.b085b44a9ffece56p-127L},
    {-0x1.4b8845d4d3bce1dcp-15L, 0x1.fffc68cf5119c93ep-1L, -0x1.fffffffffffffff6p-66L},
    {-0x1.5ab466bb2121b42ep-15L, 0x1.fffc3ebe8713b664p-1L, 0x1.4794c429a7c346fep-127L},
    {-0x1.7c6bb99fdcfc558ap-15L, 0x1.fffbe14422b12992p-1L, 0x1.e4881d91d2510f54p-127L},
    {-0x1.96b881a89db711a6p-15L, 0x1.fffb9859824377a6p-1L, 0x1.32dbd19e16999c6ap-126L},
    {-0x1.d8b40a96c0713da2p-15L, 0x1.fffae169ee695a74p-1L, -0x1.11d73a958b1b6542p-127L},
    {-0x1.e72e5f9c812b5b42p-15L, 0x1.fffab9463445011ap-1L, -0x1.72ea717dcf13acaap-128L},
    {-0x1.5ddcb2f8e0723232p-14L, 0x1.fff86c023545a07ap-1L, -0x1.b50115bd78aebdb2p-129L},
    {-0x1.67ea37c6de6a2772p-14L, 0x1.fff834446d42642ap-1L, 0x1.6820caff6c7a57d2p-133L},
    {-0x1.6ba69f937e83e83ap-14L, 0x1.fff81f8d4e4c103cp-1L, 0x1.99ebbd1392e0286ap-128L},
    {-0x1.7ce1f43c7d85bb2ep-14L, 0x1.fff7c00132fd1274p-1L, 0x1.6d7e7a0efa77c39p-128L},
    {-0x1.8ab5703915d7e7b6p-14L, 0x1.fff77357636a03ap-1L, -0x1.29904853f06bd67ep-128L},
    {-0x1.958e70bf63affe9ap-13L, 0x1.ffee6e89752e81ap-1L, 0x1.b5b6383b18c01722p-126L},
    {-0x1.9c58f6ff042fa28ap-13L, 0x1.ffee233b3604cefap-1L, 0x1.2257cc77e2096b9ep-128L},
    {-0x1.d92ae26a88c884dep-12L, 0x1.ffd702732609618cp-1L, 0x1.a1fb5ae0eb0555a8p-127L},
    {-0x1.ea3acc97462ccf9ap-12L, 0x1.ffd5881e8b4652e2p-1L, 0x1.26c68ec1d499c84cp-128L},
    {-0x1.42978fbd5903d9c8p-11L, 0x1.ffc81c69192eabdap-1L, -0x1.fffffffffffffffep-66L},
    {-0x1.e39e86a4effe162cp-11L, 0x1.ffac38d23251f566p-1L, 0x1.f2e25b6e60a416eap-128L},
    {-0x1.ec5b7502dd091e32p-11L, 0x1.ffaab570c1cc7738p-1L, -0x1.fdce3e329720149ep-129L},
    {-0x1.ef4f6db7dd41df6p-11L, 0x1.ffaa3286acf30704p-1L, -0x1.2803d5a97386f78p-127L},
    {-0x1.b6270cdcb5d62e2p-10L, 0x1.ff683c5f631723d4p-1L, 0x1.efc1c6c83b14ee7cp-126L},
    {-0x1.ce77a76194b6190ap-6L, 0x1.f61475edd297533ep-1L, 0x1.fffffffffffffffcp-66L},
    {-0x1.c8bc4afa684dfb16p-5L, 0x1.ec9744ec13971318p-1L, 0x1.45c9c7b122046e0ap-128L},
    {-0x1.9606bc444cdfdb8ep-11L, 0x1.ffb9a8ee14823cecp-1L, 0x1.670f6e9e6bffad8ep-127L},
    {-0x1.4a8794b3444605b6p-4L, 0x1.e425e2a6cf326f64p-1L, 0x1.f665a12d037c9a9ep-127L},
    {-0x1.c2395606661e0442p-7L, 0x1.fb25a3b00bcab8acp-1L, -0x1.bb0595f4e32cef44p-127L},
    {-0x1.b9e793210ea77cd2p-9L, 0x1.fece0d7e315971e2p-1L, 0x1.27a063fabdc6a27ap-128L},
    {-0x1.0611463d175f85ap-8L, 0x1.fe95334b2831567ep-1L, -0x1.f190e19ec9145434p-130L},
    {-0x1.662ae8d7db1d7b64p-10L, 0x1.ff83ed651f45aadp-1L, 0x1.9cc5805730f3e0bap-127L},
    {-0x1.321d2989bbf9a8eep-7L, 0x1.fcb2039343b4a4d6p-1L, -0x1.544567604e3a3b9p-126L},
    {-0x1.80b57cdb68b5d27ep-6L, 0x1.f7bbfcb382ac4c1ep-1L, 0x1.9be26feb8ec208f6p-130L},
    {-0x1.e4e518f1dc4e4e3ep-9L, 0x1.feb053b79bb6bb94p-1L, 0x1.af7462efb3751982p-129L},
    {-0x1.74e0d86793c6427p-9L, 0x1.fefdcb8cb398fcacp-1L, 0x1.43f5fb2e40d9e5dep-126L},
    {-0x1.02c6e20796e48112p-13L, 0x1.fff4ca31fa11d54ap-1L, -0x1.45c798e6c389e552p-126L},
    {-0x1.2c6c02eaf7d1b836p-6L, 0x1.f988a623c6d84c8p-1L, 0x1.c36863665b9fa9f4p-129L},
    {-0x1.1478d76251cf60c6p-5L, 0x1.f429699bc0bf4ca6p-1L, 0x1.640aee999363f43ap-125L},
    {-0x1.44779ac3b17ba42cp-10L, 0x1.ff8f98b19fa02628p-1L, -0x1.0e9fe02acd63be4p-129L},
    {-0x1.3e8ea85e774dbeb6p-9L, 0x1.ff2360fa55a6ded6p-1L, 0x1.04365dbd66c73ca4p-125L},
    {-0x1.4309583c53a5b05p-9L, 0x1.ff20478094da34cep-1L, 0x1.097b8af38de837c8p-125L},
    {-0x1.5e0e3ccfed32c13ep-8L, 0x1.fe1b9dfb49e09ad2p-1L, 0x1.93e5aac650ef4d46p-125L},
    {-0x1.b8654bd94a95d0b6p-11L, 0x1.ffb3b51bc1dd03dep-1L, 0x1.b2518094d9dbf2bcp-127L},
    {-0x1.218bf3fadd1d1dc4p-9L, 0x1.ff377481b5e56918p-1L, 0x1.8f2a1a9d88da9f82p-129L},
    {-0x1.938fc9609a1d4edap-5L, 0x1.eecfe8bd94c212a6p-1L, 0x1.98eb9b668e3753b4p-128L},
    {-0x1.6694a2efb4b2a60cp-11L, 0x1.ffc1e0a000f69c52p-1L, -0x1.1b25fea9a85b81a6p-127L},
    {-0x1.2771f9c4f8e81c2p-10L, 0x1.ff99a57f5f0ca2aep-1L, -0x1.59461c3100b07cd8p-128L},
    {-0x1.e215318af03341a6p-5L, 0x1.eb891af3c386ff94p-1L, 0x1.0345ce2bf15a52b2p-125L},
    {-0x1.009a4503c87f2d6ep-14L, 0x1.fffa711f85aa0ed8p-1L, -0x1.9ee76149ec65c2d2p-130L},
    {-0x1.45610e6cbd8dbbdap-5L, 0x1.f218a6c2e13d511p-1L, -0x1.93da1a61f13243d6p-126L},
    {-0x1.0ab19befd67fc6f8p-5L, 0x1.f49364ab7e379fc8p-1L, -0x1.c9f37735530e3728p-128L},
    {-0x1.604925c5bf3474d6p-3L, 0x1.c673d28ce068abb6p-1L, -0x1.d81e0bc58e392faap-127L},
    {-0x1.b78bf7500259df88p-12L, 0x1.ffd9ebf51ffc3274p-1L, 0x1.5ae21082c71c26b6p-128L},
    {-0x1.333f5952d8506128p-6L, 0x1.f9634897f4b3e7bcp-1L, -0x1.91eaaa3b4d5ed45ep-131L},
    {-0x1.364995dca344ad7p-7L, 0x1.fca68486e6ac91f4p-1L, 0x1.e1914d127a782668p-128L},
    {-0x1.af3968477ccaa31ep-9L, 0x1.fed5704d93985996p-1L, 0x1.62949b8a9040d598p-126L},
    {-0x1.0924dceecfaa488cp-11L, 0x1.ffd20fe5b28de0dcp-1L, 0x1.ad26ed72b8e4f21ap-129L},
    {-0x1.24c0df7eef4cfac2p-14L, 0x1.fff9a8ab6eed77e2p-1L, -0x1.10f21e626f315944p-128L},
    {-0x1.8505d93927bb4c3ep-4L, 0x1.df6117f984663e66p-1L, 0x1.7d207c56bc9eedbap-127L},
    {-0x1.5e2587ada2074bcep-4L, 0x1.e28b1e5c682df14ap-1L, -0x1.620d31ca3c9bf802p-130L},
    {-0x1.23ad39ace8276538p-14L, 0x1.fff9aea3e0070826p-1L, -0x1.138a52de5e238772p-128L},
    {-0x1.9b9cf2aa73d238d2p-10L, 0x1.ff716c629b6cc1f6p-1L, -0x1.0a165b72ba6cb82p-127L},
    {-0x1.533bd3dbd08af5a4p-13L, 0x1.fff14dfd1fed1ep-1L, 0x1.345c8fa0817844ecp-126L},
    {-0x1.2bfe8ec84fe88d2cp-9L, 0x1.ff30397fe83cdbacp-1L, 0x1.067355a305e4ca26p-125L},
    {-0x1.06777344730121ap-11L, 0x1.ffd286a0fa3d2996p-1L, -0x1.e039c7f93b50b22cp-129L},
    {-0x1.3156bf341603b4ep-3L, 0x1.cdbb2250ecf28d18p-1L, 0x1.51f7c471f44bbd42p-126L},
    {-0x1.a1e464cd5bc4a23ap-3L, 0x1.bc7904bc8f246052p-1L, -0x1.ef0ffdad209a7e62p-128L},
    {-0x1.9ed07fa75f06bea6p-3L, 0x1.bcef9ae152269ee2p-1L, -0x1.19a488c957763c78p-128L},
    {-0x1.37b7d6ebc3c535c2p-2L, 0x1.9e9a71830d784296p-1L, -0x1.fffffffffffffffcp-66L},
    {-0x1.227c3bbe796837f8p-2L, 0x1.a49af00837c3b46ap-1L, 0x1.55129bf7e816581ap-129L},
    {-0x1.56b05bdd054d245ep-1L, 0x1.41f2cb598284c76ap-1L, 0x1.d2f63b235d1b5822p-129L},
    {-0x1.3928fef54f77ebe6p-1L, 0x1.4f145246ca66c496p-1L, 0x1.38c74600bb4d06a4p-126L},
    {-0x1.262ad7e682c0769cp+0L, 0x1.cdbb2250ecf28d18p-2L, 0x1.51f7c471f44bbd42p-127L},
    {-0x1.489f0eef9e5a0dfep+0L, 0x1.a49af00837c3b46ap-2L, 0x1.55129bf7e816581ap-130L},
    {-0x1.93156bf341603b4ep+1L, 0x1.cdbb2250ecf28d18p-4L, 0x1.51f7c471f44bbd42p-129L},
    {-0x1.05ece6c8bd30968cp-1L, 0x1.6725658526f34c7ap-1L, -0x1.977481b2530f44f6p-129L},
    {-0x1.4e9e87fd5c97e3b4p-1L, 0x1.457c21a3a033a3ecp-1L, -0x1.56dfc93184a53a02p-128L},
    {-0x1.82f673645e984b46p+0L, 0x1.6725658526f34c7ap-2L, -0x1.977481b2530f44f6p-130L},
    {-0x1.a74f43feae4bf1dap+0L, 0x1.457c21a3a033a3ecp-2L, -0x1.56dfc93184a53a02p-129L},
    {0xb.8aa3b295c17f0bcp-67L, 0x1.0000000000000002p+0L, -0x1.fffffffffffffffep-65L},
    {0xa.194f3c43094f2a2p-64L, 0x1.0000000000000006p+0L, 0x1.fffffffffffffffep-65L},
    {0xc.434dedbf1d96fc1p-63L, 0x1.0000000000000012p+0L, -0x1.fffffffffffffffep-65L},
    {0xb.6fc4ed79fcd7255p-53L, 0x1.0000000000003f6ap+0L, 0x1.fffffffffffffffep-65L},
    {0xf.49f104ab3cc2d94p-52L, 0x1.000000000000a98ep+0L, 0x1.fffffffffffffffep-65L},
    {0x9.f1ecf60af3e5853p-47L, 0x1.00000000000dc966p+0L, 0x1.fffffffffffffffep-65L},
    {0xc.3dc8cf1463af62fp-47L, 0x1.000000000010f85ap+0L, -0x1.fffffffffffffffep-65L},
    {0x9.ad1f062a8ab29ffp-40L, 0x1.0000000006b50272p+0L, 0x1.fffffffffffffffep-65L},
    {0xd.abfd779809f67b6p-38L, 0x1.0000000025e8087ap+0L, -0x1.fffffffffffffffep-65L},
    {0xc.762d7684ae1beeap-37L, 0x1.00000000451a19cep+0L, 0x1.fffffffffffffffep-65L},
    {0xe.0c9e1609da847dbp-37L, 0x1.000000004de7e1e2p+0L, 0x1.fffffffffffffffep-65L},
    {0x9.aab514ef3077eddp-36L, 0x1.000000006b3561fep+0L, -0x1.fffffffffffffffep-65L},
    {0xd.f39d71dc272a58p-29L, 0x1.0000004d5d3d3d86p+0L, -0x1.fffffffffffffffep-65L},
    {0xa.824ad65265e94b6p-25L, 0x1.000003a4626653aap+0L, 0x1.fffffffffffffffep-65L},
    {0xd.0527fc86dd2ec59p-25L, 0x1.000004832f1eead2p+0L, -0x1.fffffffffffffffep-65L},
    {0xd.ca1bcc03e818338p-25L, 0x1.000004c7714ce422p+0L, 0x1.fffffffffffffffep-65L},
    {0xc.5f396165dfc60bap-11L, 0x1.0112fe9112c95b06p+0L, 0x1.fffffffffffffffep-65L},
    {0x1.1cac23cf32997fa6p-6L, 0x1.031a0d2f944dc4d8p+0L, 0x1.fc33e05ac1b1158ap-129L},
    {0x1.248230c2bb787ce4p-16L, 0x1.0000cac0b15d6024p+0L, -0x1.ab58fc5c42eab87p-130L},
    {0x1.2574cfe96b07e51ep-15L, 0x1.000196d25dbbb85p+0L, -0x1.650ba11717cb4bbcp-130L},
    {0x1.270a4a527eb90b6cp-7L, 0x1.019a4aa31b259dccp+0L, -0x1.7e68a9c64a6a7efp-131L},
    {0x1.35e0b2e14748db7cp-7L, 0x1.01aefe25aea5272ap+0L, -0x1.80c0b33e4cf8aac2p-127L},
    {0x1.3ac9a43d4e7d192ep-5L, 0x1.06e901f58091b67ap+0L, 0x1.120ee5fe92e5b42cp-129L},
    {0x1.3f02d33da85d3b6ep-2L, 0x1.3db3eddfcd080064p+0L, 0x1.7075b144578cbff8p-129L},
    {0x1.491705f0ae9f98bep-4L, 0x1.0ea943b7cdc4830cp+0L, -0x1.97b4ec60a25776eep-126L},
    {0x1.4df4919b6022268cp-6L, 0x1.03a47e1e06af54d4p+0L, -0x1.08060332aa1ef138p-128L},
    {0x1.50919d96b5fae21p-5L, 0x1.0765299e343f756ep+0L, 0x1.c4f0626b24f2151cp-127L},
    {0x1.5178a614b366f2fap-5L, 0x1.076a4fcbe306eadp+0L, 0x1.dc18dc836e58cc56p-125L},
    {0x1.529f4845f565b744p-2L, 0x1.41f2cb598284c76ap+0L, 0x1.d2f63b235d1b5822p-128L},
    {0x1.58b0bc0151b40e26p+0L, 0x1.457c21a3a033a3ecp+1L, -0x1.56dfc93184a53a02p-126L},
    {0x1.5afc7d79dedd2a4cp-6L, 0x1.03c92571dc388a4cp+0L, 0x1.78fb4b5ddf1a16ccp-129L},
    {0x1.5ead8ebb36c52e3p-16L, 0x1.0000f312bd341228p+0L, 0x1.ef4c0926ab586534p-132L},
    {0x1.5f5b152690eba5dap-13L, 0x1.00079c717ef7efcp+0L, 0x1.313adf5b534e0502p-127L},
    {0x1.62c2f00546d03898p-2L, 0x1.457c21a3a033a3ecp+0L, -0x1.56dfc93184a53a02p-127L},
    {0x1.658382b8511ee5ccp-10L, 0x1.003dfb508259ecacp+0L, 0x1.aff6ac6986857a6cp-126L},
    {0x1.6ec1e220c34be404p-1L, 0x1.a49af00837c3b46ap+0L, 0x1.55129bf7e816581ap-128L},
    {0x1.6f9ce5a8b3243262p-7L, 0x1.01ff9b337f526032p+0L, 0x1.25f7555adb61477cp-128L},
    {0x1.70fd6310d1b4994cp-6L, 0x1.0407157c0ce85144p+0L, 0x1.0e68d791be9eb2fcp-133L},
    {0x1.7d098c9ba167b4bap+0L, 0x1.6725658526f34c7ap+1L, -0x1.977481b2530f44f6p-127L},
    {0x1.8dae021561102834p-2L, 0x1.4f145246ca66c496p+0L, 0x1.38c74600bb4d06a4p-125L},
    {0x1.a4ed7fbb4a9fb356p-4L, 0x1.12e68526b08d8282p+0L, -0x1.dbb94f6d0a942a3ap-127L},
    {0x1.aaded45884e59364p-12L, 0x1.00127ed001fc8accp+0L, -0x1.0ac20ca1ef316aeep-128L},
    {0x1.ad988d3081bcbb9cp-4L, 0x1.134dd395bd76f908p+0L, 0x1.dc94128e60787ebp-127L},
    {0x1.ae30b1e652dca39ap-12L, 0x1.0012a3a3fccb6446p+0L, 0x1.6106632122af6d9cp-129L},
    {0x1.b3aa5032fa7f12c8p-1L, 0x1.cdbb2250ecf28d18p+0L, 0x1.51f7c471f44bbd42p-125L},
    {0x1.b760f11061a5f202p+0L, 0x1.a49af00837c3b46ap+1L, 0x1.55129bf7e816581ap-127L},
    {0x1.c400323ab65060d8p-4L, 0x1.14598c62848ce032p+0L, 0x1.a574d511f0618ab2p-127L},
    {0x1.cf8852012559841ep-2L, 0x1.5e5a8e406ecbb63ap+0L, 0x1.ab1104fa34c02b38p-131L},
    {0x1.d00a4c793a1d6d4ep-16L, 0x1.000141a6b8f91d42p+0L, -0x1.b86975165f93cd9p-128L},
    {0x1.d2eb2bfd12d6f486p-4L, 0x1.150c5eb3832acc14p+0L, 0x1.2883e8680287fe9ap-128L},
    {0x1.d9d528197d3f8964p+0L, 0x1.cdbb2250ecf28d18p+1L, 0x1.51f7c471f44bbd42p-124L},
    {0x1.db4b22a09e022f6p-13L, 0x1.000a4bcb36ef561p+0L, -0x1.56ab41256e8ece16p-130L},
    {0x1.e2dda3cd8c341298p-11L, 0x1.0029d9b9a11881b8p+0L, -0x1.1422c5751fe6962cp-128L},
    {0x1.e5b7eae7259fcb4cp-5L, 0x1.0abd81e709e4f1a4p+0L, 0x1.6109741735fe354ap-127L},
    {0x1.eaab0d7de0384c5ap-3L, 0x1.2e3f3978515cbfap+0L, 0x1.57a35d3d4f378412p-126L},
    {0x1.eb990e74b7582b7p-5L, 0x1.0adf7c7d0f3e7b3p+0L, 0x1.7449760cad2f03d4p-125L},
    {0x1.ecea940cbe9fc4b2p+1L, 0x1.cdbb2250ecf28d18p+3L, 0x1.51f7c471f44bbd42p-122L},
    {0x1.f426326e859ed2e8p-2L, 0x1.6725658526f34c7ap+0L, -0x1.977481b2530f44f6p-128L},
  };
  for (int i = 0; i < EXCEPTIONS; i++)
  {
    if (x == exceptions[i][0])
    {
      *h = exceptions[i][1];
      *l = exceptions[i][2];
      return;
    }
  }
  
  int32_t k = __builtin_roundl (0x1p15L * x); // -16445*2^15 <= k <= 16383*2^15
  // if (x == TRACE) printf ("k=%d\n", k);
  long double r = x - (long double) k * 0x1p-15L;
  // if (x == TRACE) printf ("r=%La\n", r);
  int32_t i = (k + 538869760) & 32767;
  // if (x == TRACE) printf ("i=%d\n", i);
  int32_t e = (k - i) >> 15;
  // if (x == TRACE) printf ("e=%d\n", e);
  int32_t i0 = i & 0x1f, i1 = (i >> 5) & 0x1f, i2 = i >> 10;
  // if (x == TRACE) printf ("i2=%d i1=%d i0=%d\n", i2, i1, i0);
  Pacc (h, l, r);
  // if (x == TRACE) printf ("P: h=%La l=%La\n", *h, *l);
  long double hh, ll;
  d_mul (&hh, &ll, T2[i2][0], T2[i2][1], T1[i1][0], T1[i1][1]);
  d_mul (&hh, &ll, hh, ll, T0[i0][0], T0[i0][1]);
  d_mul (h, l, *h, *l, hh, ll);
  // normalize h+l
  fast_two_sum (h, l, *h, *l);
  // if (x == TRACE) printf ("x=%La h=%La l=%La e=%d\n", x, *h, *l, e);
  if (e >= -16381)
  {
    /* Since |h| > 0.5, ulp(h) >= 2^-64, thus ulp(h)*2^e >= 2^-16445 which is the smallest
       subnormal, thus 2^e*h is exact. */
    *h = __builtin_ldexpl (*h, e);
    *l = __builtin_ldexpl (*l, e);
  }
  else // near subnormal range
  {
    hh = *h;
    *h = __builtin_ldexpl (*h, e); // might not equal 2^e*h
    // if (x == TRACE) printf ("h=%La\n", *h);
    hh = hh - __builtin_ldexpl (*h, -e); // remaining (truncated) part
    // if (x == TRACE) printf ("hh=%La\n", hh);
    hh += *l;
    // if (x == TRACE) printf ("hh=%La\n", hh);
    *l = __builtin_ldexpl (hh, e);
    // if (x == TRACE) printf ("l=%La\n", *l);
  }
}

long double
cr_exp2l (long double x)
{
  b96u96_u v = {.f = x};
  uint32_t e = v.e & 0x7fff;
  // printf ("x=%La v.e=%u\n", x, v.e);

  // check NaN, Inf, overflow, underflow
  // overflow for x >= 16384, i.e., 16397 <= e <= 32767
  // the smallest subnormal is 2^-16445
  if (__builtin_expect (e >= 16397, 0))
  {
    if (e == 0x7fff)
    { // NaN or Inf: 2^x = x for x = NaN or +Inf, 2^-Inf = 0
      if (v.e == 0xffff && v.m == 0x8000000000000000ul) // -Inf
        return 0x0p0L;
      return x;
    }
    if (x >= 0x1p+14L) // x >= 16384
      return 0x1p16383L + 0x1p16383L;
    // now x < 0
    if (x <= -0x1.00f8p+14L) // x <= -16446
      return 0x1p-16445L * 0.5L;
  }

  // case of tiny inputs
  // for 0 <= x <= 0x1.71547652b82fe176p-64, 2^x rounds to 1 to nearest
  // for -0x1.71547652b82fe176p-65 <= x <= 0, 2^x rounds to 1 to nearest
  if (__builtin_expect (e <= 16319, 0)) // |x| < 2^-63
  {
    if (0 <= x && x <= 0x1.71547652b82fe176p-64L)
      return __builtin_fmal (x, x, 0x1p0L);
    if (-0x1.71547652b82fe176p-65L <= x && x < 0)
      return __builtin_fmal (x, -x, 0x1p0L);
  }

  // now -16446 < x < -0x1.71547652b82fe176p-65 or 0x1.71547652b82fe176p-64 < x < 16384

  long double h, l;
  fast_path (&h, &l, x);
  static const long double err = 0x1.26p-86; // 2^-85.803 < err
  //if (x == TRACE) printf ("h=%La l=%La\n", h, l);
  long double left = h +  (l - h * err);
  long double right = h + (l + h * err);
  //if (x == TRACE) printf ("left=%La right=%La\n", left, right);
  if (__builtin_expect (left == right, 1))
    return left;

  //if (x == TRACE) printf ("fast path failed\n");

  accurate_path (&h, &l, x);
  return h + l;
}
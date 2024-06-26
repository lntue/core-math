#!/bin/bash

set -e

if [ -z "$LAST_COMMIT" ]; then
    LAST_COMMIT="HEAD~"
    if [ -n "$CI_COMMIT_BEFORE_SHA" ] && [ "$CI_COMMIT_BEFORE_SHA" != "0000000000000000000000000000000000000000" ]; then
        LAST_COMMIT="$CI_COMMIT_BEFORE_SHA"
    fi
fi

# use the same order as on https://core-math.gitlabpages.inria.fr/
FUNCTIONS_EXHAUSTIVE=(acosf acoshf acospif asinf asinhf asinpif atanf atanhf atanpif cbrtf cosf coshf cospif erff erfcf expf exp10f exp10m1f exp2f exp2m1f expm1f lgammaf logf log10f log10p1f log1pf log2f log2p1f rsqrtf sinf sinhf sinpif tanf tanhf tanpif tgammaf)
FUNCTIONS_WORST=(acos acosh acospi asin asinh asinpi atan atan2 atan2f atan2pi atan2pif atanh atanpi cbrt cbrtl cos cosh cospi erf erfc exp exp10 exp10m1 exp2 exp2l exp2m1 hypot hypotf log log10 log10p1 log1p log2 log2p1 pow powf rsqrt sin sinh sinpi tan tanh tanpi)
FUNCTIONS_SPECIAL=(atan2pif hypotf)

echo "Reference commit is $LAST_COMMIT"

check () {
    KIND="$1"
    if ! { echo "$FORCE_FUNCTIONS" | tr ' ' '\n' | grep --quiet '^'"$FUNCTION"'$'; } && git diff --quiet "$LAST_COMMIT".. -- src/*/*/$FUNCTION.c; then
        echo "Skipped $FUNCTION"
    else
        echo "Checking $FUNCTION..."
        ./check.sh "$KIND" "$FUNCTION"
    fi
}

for FUNCTION in "${FUNCTIONS_EXHAUSTIVE[@]}"; do
    check --exhaustive
done

for FUNCTION in "${FUNCTIONS_WORST[@]}"; do
    check --worst
done

for FUNCTION in "${FUNCTIONS_SPECIAL[@]}"; do
    check --special
done

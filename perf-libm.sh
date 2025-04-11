#!/bin/bash

declare -a func_list=(
    "acosf" "acoshf" "asinf" "asinhf" "atan2" "atan2f" "atanf" "atanhf"
    "cbrt" "cbrtf"
    "cos" "cosf" "coshf" "cospif"
    "erff" "exp" "exp10" "exp10f" "exp2" "exp2f" "exp2m1f"
    "expf" "expm1" "expm1f"
    "hypotf"
    "log" "log10" "log10f" "log1p" "log1pf" "log2" "log2f" "logf"
    "pow" "powf"
    "sin" "sincos" "sincosf" "sinf" "sinhf" "sinpif"
    "tan" "tanf" "tanhf"
)

if [ -z "${HOME_PATH}" ]; then
    HOME_PATH=$HOME
fi

while [ -n "$1" ]; do
  case "$1" in
    --llvm )
      LIBM="${HOME_PATH}/llvm-project/build/projects/libc/lib/libllvmlibc.a";
      flag="--llvm --skip_system_libc --skip_core_math";
      shift ;;
    --musl )
      LIBM="${HOME_PATH}/musl/lib/libc.a";
      flag="--musl --skip_system_libc --skip_core_math";
      shift ;;
    --core_math )
      LIBM="${HOME_PATH}/core_math/";
      flag="--skip_system_libc";
      shift ;;
    --system )
      LIBM="System libc";
      flag="--skip_core_math";
      shift ;;
    --latency )
      LATENCY="--latency";
      shift ;;
    * ) break ;;
  esac
done

echo "LIBM location: ${LIBM}"

for func in "${func_list[@]}"
do
  echo -n "$func "
  CORE_MATH_QUIET=TRUE ./perf.sh $func --fast $flag
done

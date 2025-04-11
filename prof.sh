#!/bin/bash
# Usage: ./prof.sh acos

RANDOMS_FILE="$(mktemp /tmp/core-math.XXXXXX)"
PERF_DATA_FILE="$(mktemp /tmp/core-math.XXXXXX)"
trap "rm -f $RANDOMS_FILE $PERF_DATA_FILE" 0

f=$1
u="$(echo src/binary*/*/$f.c)"

if [ -f "$u" ]; then
    dir="${u%/*}"
else
    echo "Unknown function: $f"
    exit 2
fi

# for clang we might want to add -ffp-contract=on to enable FMA
if [ "$CFLAGS" == "" ]; then
   export CFLAGS="-O3 -march=native"
fi

CFLAGS="$CFLAGS -ggdb"

export LIBM="$HOME/experiment/llvm-project/build/projects/libc/lib/libllvmlibc.a"

# COMMON_PERF_ARGS=(--file "$RANDOMS_FILE" --count 10000 --libc --latency)
COMMON_PERF_ARGS=(--file "$RANDOMS_FILE" --count 100000 --libc )

cd $dir
make -s clean
make -s perf
./perf "${COMMON_PERF_ARGS[@]}" --reference

perf record --output="$PERF_DATA_FILE" ./perf "${COMMON_PERF_ARGS[@]}" --repeat 500
perf annotate --input="$PERF_DATA_FILE" --percent-type=local-hits

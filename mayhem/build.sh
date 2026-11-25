#!/usr/bin/env bash
#
# mayhem/build.sh — build cvxopt's C extensions twice and the fuzz drivers.
#
#  1. Sanitized build of every extension (base, blas, lapack, amd, cholmod, umfpack,
#     misc_solvers from src/C) with $SANITIZER_FLAGS + SanCov (-fsanitize=fuzzer-no-link)
#     + $DEBUG_FLAGS, into mayhem/out/asan-lib/cvxopt (gitignored).
#  2. One libFuzzer binary per Python harness (mayhem/harnesses/fuzz_<t>.py): the
#     CPython-embedding driver mayhem/harnesses/pyfuzz_main.c, plus its -standalone
#     reproducer. The harnesses reach cvxopt only through its public Python API.
#  3. Normal-flags build of the extensions (mayhem/out/test-lib) and a project-built
#     CPython launcher (mayhem/out/cvxopt-python) that mayhem/test.sh runs pytest with.
#
# Dependencies (BLAS/LAPACK, SuiteSparse, Python headers, setuptools, pytest) come from
# apt in mayhem/Dockerfile, so re-running this offline works (setup.py build is
# incremental and never fetches: no pip, no build isolation).
set -euo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
: "${SRC:=/mayhem}"
: "${STANDALONE_FUZZ_MAIN:=/opt/mayhem/StandaloneFuzzTargetMain.c}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"
OUT="$SRC/mayhem/out"
mkdir -p "$OUT"

# setuptools_scm would otherwise ask git for the version; pin it so the build is
# independent of the checkout's tags.
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-1.3.3}"

# The fuzz flags only apply when sanitizers are on; with SANITIZER_FLAGS= (empty) the
# extensions are still built with SanCov-free debug info.
SANCOV=""
[ -n "$SANITIZER_FLAGS" ] && SANCOV="-fsanitize=fuzzer-no-link"

# 1) sanitized extensions (CFLAGS are appended after Python's own, so -gdwarf-3 wins)
# -fno-sanitize=function: cvxopt's C-API export table (base.c API[] capsule, called as
# e.g. Matrix_Check through int (*)(void *)) and its per-typecode function tables
# (scal[id], write_num[id], ...) call through deliberately generic pointer types; that
# UBSan check fires on every input and hides everything else. ASan and the rest of
# UBSan stay on and halting.
CFLAGS="$SANITIZER_FLAGS -fno-sanitize=function $SANCOV $DEBUG_FLAGS -O1" \
LDSHARED="$CC -shared" LDFLAGS="$SANITIZER_FLAGS $SANCOV" \
  python3 setup.py -q build -j "$MAYHEM_JOBS" \
    --build-base "$OUT/asan-build" --build-lib "$OUT/asan-lib"

# 2) fuzz drivers
PYINC="$(python3-config --includes)"
PYLD="$(python3-config --ldflags --embed)"
$CXX $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o "$OUT/lsan_off.o"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$OUT/standalone_main.o"
for py in "$SRC"/mayhem/harnesses/fuzz_*.py; do
  t="$(basename "$py" .py)"
  defs=(-DPYFUZZ_MODULE="\"$t\"" -DPYFUZZ_LIBDIR="\"$OUT/asan-lib\""
        -DPYFUZZ_HARNESS_DIR="\"$SRC/mayhem/harnesses\"")
  $CC $SANITIZER_FLAGS $DEBUG_FLAGS $PYINC "${defs[@]}" -c "$SRC/mayhem/harnesses/pyfuzz_main.c" \
      -o "$OUT/$t.o"
  # -rdynamic: the dlopen'ed extensions resolve the sanitizer/SanCov callbacks from here
  $CC $SANITIZER_FLAGS $DEBUG_FLAGS $LIB_FUZZING_ENGINE -rdynamic "$OUT/$t.o" "$OUT/lsan_off.o" \
      $PYLD -lstdc++ -o "$SRC/$t"
  $CC $SANITIZER_FLAGS $DEBUG_FLAGS -rdynamic "$OUT/standalone_main.o" "$OUT/$t.o" "$OUT/lsan_off.o" \
      $PYLD -lstdc++ -o "$SRC/$t-standalone"
done

# 3) normal-flags build for the functional suite + the launcher test.sh uses
env -u LDFLAGS CFLAGS="-O2 $COVERAGE_FLAGS" LDSHARED="$CC -shared $COVERAGE_FLAGS" \
  python3 setup.py -q build -j "$MAYHEM_JOBS" \
    --build-base "$OUT/test-build" --build-lib "$OUT/test-lib"
$CC -O2 $COVERAGE_FLAGS $PYINC "$SRC/mayhem/harnesses/pyrun.c" $PYLD -o "$OUT/cvxopt-python"
echo "build.sh: done"

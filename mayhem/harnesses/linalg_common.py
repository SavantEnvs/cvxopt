"""Shared driver for the BLAS/LAPACK wrapper harnesses: call a fuzz-chosen
routine with fuzz-built matrices and fuzz-chosen size/offset/stride/option
keywords, so the wrappers' own argument validation (n, m, ld*, inc*, offset*)
is what stands between the input and the Fortran library."""
from fdp import FDP, guarded, rand_matrix
from cvxopt import matrix

INT_KW = ['n', 'm', 'k', 'kl', 'ku', 'nrhs', 'ldA', 'ldB', 'ldC', 'ldW', 'ldZ', 'ldU', 'ldVt',
          'ldVl', 'ldVr', 'incx', 'incy', 'offsetA', 'offsetB', 'offsetC', 'offsetx', 'offsety',
          'offsetW', 'offsetS', 'offsetZ', 'offsetU', 'offsetVt', 'offsettau', 'offsetipiv',
          'il', 'iu', 'r', 'offsetd', 'offsete', 'offsetdl', 'offsetdu', 'offsetdu2']
CHR_KW = {'uplo': 'LUX', 'trans': 'NTCX', 'transA': 'NTC', 'transB': 'NTC', 'side': 'LRX',
          'diag': 'NUX', 'jobz': 'NVX', 'jobu': 'NASOX', 'jobvt': 'NASOX', 'jobvl': 'NVX',
          'jobvr': 'NVX', 'range': 'AVIX', 'jobv': 'NV'}
FLT_KW = ['alpha', 'beta', 'vl', 'vu', 'abstol']


def operand(f, kind):
    if kind == 'M':          # general numeric matrix
        return rand_matrix(f, f.pick('dddz'))
    if kind == 'I':          # output pivot vector: fuzzed length, contents written by LAPACK
        return matrix(0, (f.int_in(0, 8), 1), 'i')
    if kind == 'K':          # integer positional argument (m, kl)
        return f.small_int()
    if kind == 'S':          # scalar
        return f.scalar(f.pick('dz'))
    if kind == 'F':          # select callback for gees/gges
        return f.pick([None, lambda *a: True, lambda *a: 0])
    return None


def call(module, table, data):
    f = FDP(data)
    name, sig = table[f.int_in(0, len(table) - 1)]
    fn = getattr(module, name, None)
    if fn is None:
        return
    args = [operand(f, k) for k in sig]
    kw = {}
    for _ in range(f.int_in(0, 4)):
        c = f.int_in(0, 2)
        if c == 0:
            kw[f.pick(INT_KW)] = f.small_int()
        elif c == 1:
            key = f.pick(list(CHR_KW))
            kw[key] = f.pick(CHR_KW[key])
        else:
            kw[f.pick(FLT_KW)] = f.double()
    guarded(fn, *args, **kw)
    # results land in-place in the operands: read them back
    for a in args:
        if isinstance(a, matrix):
            guarded(list, a)

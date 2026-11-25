"""cvxopt.lapack.* (src/C/lapack.c): LAPACK wrappers (factorizations, solvers,
eigen/singular value routines) with fuzzed n/nrhs/ld*/offset*/kl/ku keywords.

Routines that consume a pivot vector (getrs, getri, gbtrs, gttrs, sytrs,
sytri, hetrs, hetri) get it the way a real caller does: from the matching
factorization of the same operands, called with the same n/ld/offset/uplo/kl
keywords. Only sizes, offsets and keywords are fuzzed, never pivot contents
(LAPACK documents ipiv as the *trf output and swaps rows at ipiv[k] unchecked)."""
from cvxopt import lapack, matrix
from fdp import FDP, guarded, EXPECTED
from linalg_common import call, operand

TABLE = [
    ('getrf', 'MI'), ('gesv', 'MMI'), ('gbtrf', 'MKKI'), ('gbsv', 'MKMI'),
    ('gttrf', 'MMMMI'), ('gtsv', 'MMMM'),
    ('potrf', 'M'), ('potrs', 'MM'), ('potri', 'M'), ('posv', 'MM'), ('pbtrf', 'M'),
    ('pbtrs', 'MM'), ('pbsv', 'MM'), ('pttrf', 'MM'), ('pttrs', 'MMM'), ('ptsv', 'MMM'),
    ('sytrf', 'MI'), ('hetrf', 'MI'), ('sysv', 'MMI'), ('hesv', 'MMI'), ('trtrs', 'MM'),
    ('trtri', 'M'), ('tbtrs', 'MM'), ('gels', 'MM'), ('geqrf', 'MM'), ('ormqr', 'MMM'),
    ('unmqr', 'MMM'), ('orgqr', 'MM'), ('ungqr', 'MM'), ('gelqf', 'MM'), ('ormlq', 'MMM'),
    ('unmlq', 'MMM'), ('orglq', 'MM'), ('unglq', 'MM'), ('geqp3', 'MIM'), ('syev', 'MM'),
    ('heev', 'MM'), ('syevx', 'MM'), ('heevx', 'MM'), ('syevd', 'MM'), ('heevd', 'MM'),
    ('syevr', 'MM'), ('heevr', 'MM'), ('sygv', 'MMM'), ('hegv', 'MMM'), ('gesvd', 'MM'),
    ('gesdd', 'MM'), ('gees', 'MMMF'), ('gges', 'MMMMMF'), ('trsyl', 'MMM'), ('lacpy', 'MM'),
]

# solve/inverse routine -> (factorization, keywords both calls share)
PIVOTED = [
    ('getrs', 'getrf', ['n', 'ldA', 'offsetA']),
    ('getri', 'getrf', ['n', 'ldA', 'offsetA']),
    ('gbtrs', 'gbtrf', ['n', 'ku', 'ldA', 'offsetA']),
    ('gttrs', 'gttrf', ['n', 'offsetdl', 'offsetd', 'offsetdu']),
    ('sytrs', 'sytrf', ['uplo', 'n', 'ldA', 'offsetA']),
    ('sytri', 'sytrf', ['uplo', 'n', 'ldA', 'offsetA']),
    ('hetrs', 'hetrf', ['uplo', 'n', 'ldA', 'offsetA']),
    ('hetri', 'hetrf', ['uplo', 'n', 'ldA', 'offsetA']),
]
B_KW = ['nrhs', 'ldB', 'offsetB']


def factor_then_solve(f):
    solve, factor, shared = PIVOTED[f.int_in(0, len(PIVOTED) - 1)]
    kw = {}
    for key in shared:
        if f.bool():
            kw[key] = f.pick('LU') if key == 'uplo' else f.small_int()
    ipiv = operand(f, 'I')
    if factor == 'gttrf':
        dl, d, du = (operand(f, 'M') for _ in range(3))
        du2 = operand(f, 'M')
        fargs, sargs = (dl, d, du, du2, ipiv), lambda B: (dl, d, du, du2, ipiv, B)
    else:
        A = operand(f, 'M')
        if factor == 'gbtrf':
            kl = f.int_in(0, 3)
            # m = n: gbtrs solves with the n x n band factor
            m = kw.get('n', A.size[1])
            fargs, sargs = (A, m, kl, ipiv), lambda B: (A, kl, ipiv, B)
        else:
            if factor == 'getrf':
                # getrs/getri need the LU factor of an n x n matrix
                n = kw.setdefault('n', min(A.size))
                kw['m'] = n
            fargs, sargs = (A, ipiv), lambda B: (A, ipiv, B)
    try:
        getattr(lapack, factor)(*fargs, **kw)
    except EXPECTED:
        return       # no valid factorization, so no valid pivots: a caller stops here
    kw.pop('m', None)
    if solve.endswith('tri'):
        guarded(getattr(lapack, solve), A, ipiv, **kw)
    else:
        B = operand(f, 'M')
        for key in B_KW:
            if f.bool():
                kw[key] = f.small_int()
        if solve in ('getrs', 'gbtrs', 'gttrs') and f.bool():
            kw['trans'] = f.pick('NTCX')
        guarded(getattr(lapack, solve), *sargs(B), **kw)
    for a in fargs:
        if isinstance(a, matrix):
            guarded(list, a)


def TestOneInput(data):
    if data[:1] and data[0] & 1:
        factor_then_solve(FDP(data[1:]))
    else:
        call(lapack, TABLE, data)

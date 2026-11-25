"""cvxopt.amd, cvxopt.cholmod and cvxopt.umfpack (src/C/amd.c, cholmod.c,
umfpack.c) on fuzz-built sparse matrices: ordering, symbolic/numeric
factorization, solves and factor inspection with fuzzed options."""
from fdp import FDP, guarded, rand_matrix, rand_spmatrix
from cvxopt import matrix, spmatrix, amd, cholmod, umfpack


def same_pattern(f, A):
    """A matrix with A's sparsity pattern, size and type but fuzzed values."""
    return spmatrix([f.scalar(A.typecode) for _ in range(len(A))], A.I, A.J, A.size, A.typecode)


def TestOneInput(data):
    f = FDP(data)
    n = f.dim(1, 8)
    tc = f.pick('dz')
    A = rand_spmatrix(f, tc, n, n if f.u8() < 230 else f.dim(1, 8))
    if f.bool():   # make it diagonally dominant often, so factorizations succeed
        A = A + spmatrix([10.0 + f.double() for _ in range(n)], range(n), range(n), (A.size[0], A.size[1])) \
            if A.size[0] == A.size[1] else A
    nrhs = f.dim(0, 3)
    B = rand_matrix(f, tc, A.size[0] if f.u8() < 230 else f.dim(0, 8), nrhs)
    op = f.int_in(0, 10)
    uplo = f.pick('LUX')
    if op == 0:
        guarded(amd.order, A, uplo)
    elif op == 1:
        guarded(cholmod.linsolve, A, B, uplo=uplo, nrhs=f.small_int() if f.bool() else nrhs)
    elif op == 2:
        guarded(cholmod.splinsolve, A, rand_spmatrix(f, tc, A.size[0], f.dim(1, 4)), uplo=uplo)
    elif op in (3, 4, 5):
        F = guarded(cholmod.symbolic, A, uplo=uplo) if f.bool() else \
            guarded(cholmod.symbolic, A, p=matrix([f.small_int() for _ in range(n)], tc='i'), uplo=uplo)
        if F is None:
            return
        if guarded(cholmod.numeric, A, F) is None and f.bool():
            return
        sysno = f.int_in(-1, 9)
        if op == 3:
            guarded(cholmod.solve, F, B, sys=sysno, nrhs=nrhs, ldB=f.small_int() if f.bool() else max(1, B.size[0]),
                    offsetB=f.small_int() if f.bool() else 0)
        elif op == 4:
            guarded(cholmod.spsolve, F, rand_spmatrix(f, tc, A.size[0], f.dim(1, 4)), sys=sysno)
        else:
            guarded(cholmod.diag, F)
            guarded(cholmod.getfactor, F)
    elif op == 6:
        guarded(umfpack.linsolve, A, B, trans=f.pick('NTCX'), nrhs=f.small_int() if f.bool() else nrhs,
                ldB=f.small_int() if f.bool() else max(1, B.size[0]), offsetB=f.small_int() if f.bool() else 0)
    elif op in (7, 8):
        Fs = guarded(umfpack.symbolic, A)
        if Fs is None:
            return
        Fn = guarded(umfpack.numeric, A, Fs)
        if Fn is None:
            return
        if op == 7:
            guarded(umfpack.solve, A, Fn, B, trans=f.pick('NTCX'), nrhs=f.small_int() if f.bool() else nrhs)
        else:
            # refactor/solve with new values on the same pattern and size, which is
            # the documented reuse of a symbolic/numeric factor
            A2 = same_pattern(f, A)
            guarded(umfpack.solve, A2, Fn, B)
            guarded(umfpack.numeric, A2, Fs)
    elif op == 9:
        F = guarded(cholmod.symbolic, A)
        if F is not None:
            guarded(cholmod.numeric, same_pattern(f, A), F)
            guarded(cholmod.solve, F, B)
    else:
        guarded(cholmod.options.__setitem__, 'supernodal', f.int_in(0, 2))
        guarded(cholmod.linsolve, A, B)
        cholmod.options.clear()
    cholmod.options.clear()

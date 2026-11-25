"""cvxopt.spmatrix(V, I, J, size, tc), cvxopt.sparse(blocks) and cvxopt.spdiag():
sparse construction (src/C/sparse.c spmatrix_new / SpMatrix_NewFromIJV, triplet
validation, duplicate merging, CCS build) and block concatenation (base.c)."""
from fdp import FDP, guarded, small_enough, rand_matrix, rand_spmatrix
from cvxopt import matrix, spmatrix, sparse, spdiag


def idx_list(f, n):
    kind = f.int_in(0, 3)
    # indices may be negative or out of range, but stay small: the size is inferred
    # from max(I)+1 / max(J)+1, and a huge one is an OOM, not a bug
    vals = [f.int_in(-3, 40) for _ in range(n)]
    if kind == 0:
        return vals
    if kind == 1:
        return matrix(vals, tc='i') if n else matrix([], tc='i')
    if kind == 2:
        return f.int_in(-3, 40)                    # scalar index
    return matrix([float(v) for v in vals]) if n else []  # wrong typecode


def TestOneInput(data):
    f = FDP(data)
    op = f.int_in(0, 3)
    if op == 0:
        n = f.int_in(0, 20)
        tc = f.pick('idz')
        if f.bool():
            V = [f.scalar(tc) for _ in range(n)]
        elif f.bool():
            V = f.scalar(tc)
        else:
            V = rand_matrix(f, tc, n, 1)
        I = idx_list(f, n)
        J = idx_list(f, n)
        args = [V, I, J]
        kw = {}
        if f.bool():
            args.append((f.size_int(), f.size_int()))
        if f.bool():
            kw['tc'] = f.pick('idzx')
        A = guarded(spmatrix, *args, **kw)
    elif op == 1:
        rows = []
        for _ in range(f.int_in(0, 3)):
            col = []
            for _ in range(f.int_in(0, 3)):
                c = f.int_in(0, 3)
                col.append(rand_spmatrix(f) if c == 0 else rand_matrix(f, f.pick('dz'))
                           if c == 1 else f.double() if c == 2 else [f.double()])
            rows.append(col)
        A = guarded(sparse, rows, tc=f.pick('dz')) if f.bool() else guarded(sparse, rows)
    elif op == 2:
        items = []
        for _ in range(f.int_in(0, 4)):
            c = f.int_in(0, 2)
            items.append(rand_spmatrix(f) if c == 0 else rand_matrix(f, f.pick('dz'))
                         if c == 1 else f.double())
        A = guarded(spdiag, items if f.bool() else matrix([f.double() for _ in items] or [0.0]))
    else:
        A = guarded(sparse, rand_matrix(f, f.pick('idz')))
    if A is None:
        return
    guarded(lambda: (A.size, A.V, A.I, A.J, A.CCS, A.typecode))
    if not small_enough(A):
        return
    guarded(matrix, A)
    guarded(lambda: A.T * A)
    guarded(lambda: A + A)
    guarded(lambda: sparse(A, tc='z'))

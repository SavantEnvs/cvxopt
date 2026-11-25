"""cvxopt.solvers.lp / qp / conelp / coneqp on small fuzz-built problems
(src/python/coneprog.py, misc.py) and the misc_solvers C helpers they drive
(src/C/misc_solvers.c: scale, sprod, sinv, pack/unpack, max_step, ...)."""
from fdp import FDP, guarded, rand_matrix
from cvxopt import matrix, solvers, misc_solvers


def dims_from(f):
    # coneprog rejects 'q' sizes < 1 before it reaches misc_solvers, so never draw 0
    return {'l': f.int_in(0, 4), 'q': [f.int_in(1, 4) for _ in range(f.int_in(0, 2))],
            's': [f.int_in(0, 3) for _ in range(f.int_in(0, 2))]}


def cone_rows(d):
    return d['l'] + sum(d['q']) + sum(k * k for k in d['s'])


def TestOneInput(data):
    f = FDP(data)
    solvers.options.clear()
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = f.int_in(1, 8)
    op = f.int_in(0, 4)
    n = f.dim(1, 5)
    if op == 4:
        # misc_solvers is coneprog's internal helper module: it trusts that the
        # vectors match dims (coneprog guarantees it), so only consistent calls are made
        d = dims_from(f)
        N = cone_rows(d)
        x = rand_matrix(f, 'd', N, 1)
        y = rand_matrix(f, 'd', N, 1)
        fn = f.int_in(0, 6)
        if fn == 0:
            guarded(misc_solvers.sprod, x, y, d)
        elif fn == 1:
            guarded(misc_solvers.sinv, x, y, d)
        elif fn == 2:
            guarded(misc_solvers.max_step, x, d)
        elif fn == 3:
            guarded(misc_solvers.pack, x, y, d)
        elif fn == 4:
            guarded(misc_solvers.unpack, x, y, d)
        elif fn == 5:
            guarded(misc_solvers.symm, x, int(N ** 0.5))
        else:
            guarded(misc_solvers.trisc, x, d)
            guarded(misc_solvers.triusc, x, d)
        return
    d = dims_from(f) if op == 2 else {'l': f.dim(0, 6), 'q': [], 's': []}
    m = cone_rows(d)
    c = rand_matrix(f, 'd', n, 1)
    G = rand_matrix(f, 'd', m if f.u8() < 240 else f.dim(0, 8), n)
    h = rand_matrix(f, 'd', G.size[0], 1)
    A = b = None
    if f.bool():
        p = f.dim(0, 2)
        A = rand_matrix(f, 'd', p, n)
        b = rand_matrix(f, 'd', p, 1)
    if op == 0:
        guarded(solvers.lp, c, G, h, A, b)
    elif op == 1:
        P = rand_matrix(f, 'd', n, n)
        P = P * P.T
        guarded(solvers.qp, P, c, G, h, A, b)
    elif op == 2:
        guarded(solvers.conelp, c, G, h, d, A, b)
    else:
        P = rand_matrix(f, 'd', n, n)
        guarded(solvers.coneqp, P * P.T, c, G, h, d, A, b)
    solvers.options.clear()

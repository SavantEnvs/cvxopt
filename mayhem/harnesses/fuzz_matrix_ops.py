"""matrix/spmatrix indexing, slice/index-list assignment and arithmetic
(src/C/dense.c matrix_subscr/matrix_ass_subscr, src/C/sparse.c
spmatrix_subscr/spmatrix_ass_subscr, sparse add/mul, base.c elementwise ops),
driven by a fuzz-chosen op sequence."""
from fdp import FDP, guarded, small_enough, rand_matrix, rand_spmatrix
import cvxopt
from cvxopt import matrix, spmatrix, sparse


def index(f, A):
    k = f.int_in(0, 6)
    if k == 0:
        return f.small_int()
    if k == 1:
        return slice(f.small_int() if f.bool() else None, f.small_int() if f.bool() else None,
                     (f.small_int() or 1) if f.bool() else None)
    if k == 2:
        return [f.small_int() for _ in range(f.int_in(0, 6))]
    if k == 3:
        n = f.int_in(0, 6)
        return matrix([f.small_int() for _ in range(n)], tc='i') if n else matrix([], tc='i')
    if k == 4:
        return (index1(f), index1(f))
    if k == 5:
        return matrix([f.double() for _ in range(2)])
    return (f.small_int(), f.small_int())


def index1(f):
    k = f.int_in(0, 3)
    if k == 0:
        return f.small_int()
    if k == 1:
        return slice(f.small_int() if f.bool() else None, f.small_int() if f.bool() else None,
                     (f.small_int() or 1) if f.bool() else None)
    if k == 2:
        return [f.small_int() for _ in range(f.int_in(0, 5))]
    n = f.int_in(0, 5)
    return matrix([f.small_int() for _ in range(n)], tc='i') if n else matrix([], tc='i')


def operand(f):
    k = f.int_in(0, 3)
    if k == 0:
        return rand_matrix(f)
    if k == 1:
        return rand_spmatrix(f)
    if k == 2:
        return f.scalar(f.pick('idz'))
    return [f.double() for _ in range(f.int_in(0, 4))]


def step(f, A):
    op = f.int_in(0, 16)
    if op == 0:
        return A, A[index(f, A)]
    if op == 1:
        A[index(f, A)] = operand(f)
        return A, None
    if op == 2:
        return A + operand(f), None
    if op == 3:
        return A * operand(f), None
    if op == 4:
        B = A
        B += operand(f)
        return B, None
    if op == 5:
        B = A
        B -= operand(f)
        return B, None
    if op == 6:
        return A.T, A.ctrans()
    if op == 7:
        return operand(f) * A, None
    if op == 8:
        return A / operand(f), None
    if op == 9:
        A.size = (f.size_int(), f.size_int())
        return A, None
    if op == 10:
        return A, (cvxopt.mul(A, operand(f)), cvxopt.div(A, operand(f)))
    if op == 11:
        return A, (cvxopt.max(A), cvxopt.min(A), sum(A) if len(A) < 200 else 0)
    if op == 12:
        return A, (abs(A), -A, +A, A ** f.pick([1, 2, 0.5, -1, 3]))
    if op == 13:
        return A, (cvxopt.exp(A), cvxopt.log(A), cvxopt.sqrt(A), cvxopt.cos(A), cvxopt.sin(A))
    if op == 14:
        return A, A.real() if hasattr(A, 'real') else None
    if op == 15:
        return A, (A == A, list(A), A.imag() if hasattr(A, 'imag') else None)
    return (matrix(A) if isinstance(A, spmatrix) and small_enough(A) else sparse(A)), None


def TestOneInput(data):
    f = FDP(data)
    A = guarded(rand_matrix, f) if f.bool() else guarded(rand_spmatrix, f)
    if A is None:
        return
    for _ in range(f.int_in(1, 10)):
        if f.remaining() <= 0:
            break
        r = guarded(step, f, A)
        if r is not None and r[0] is not None and isinstance(r[0], (matrix, spmatrix)):
            if len(r[0]) <= 400:
                A = r[0]

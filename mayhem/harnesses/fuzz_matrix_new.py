"""cvxopt.matrix(x, size, tc): the dense constructor (src/C/dense.c matrix_new),
fed buffer-protocol exporters (Matrix_NewFromPyBuffer, upstream fix 0584e21),
nested sequences (Matrix_NewFromSequence), numbers and other matrices."""
import array
import struct
from fdp import FDP, EXPECTED, guarded, small_enough
import cvxopt
from cvxopt import matrix, spmatrix

try:
    import numpy
except ImportError:          # numpy is optional; array/memoryview still reach the code
    numpy = None

ARRAY_FMT = 'bBhHiIlLqQfd'
NP_DT = ['i1', 'u1', 'i2', 'i4', 'i8', 'u8', 'f4', 'f8', 'c8', 'c16', '?', '>f8', '<i4']


def make_obj(f):
    kind = f.int_in(0, 7)
    n = f.int_in(0, 24)
    if kind == 0:                       # 1-D array.array
        fmt = f.pick(ARRAY_FMT)
        a = array.array(fmt, bytes(f.bytes(n * array.array(fmt).itemsize)))
        return a
    if kind == 1:                       # memoryview.cast to 1-D/2-D shape
        fmt = f.pick('bBhHiIlqdfc')
        it = struct.calcsize(fmt)
        raw = bytes(f.bytes(n * it))
        mv = memoryview(raw)
        try:
            if f.bool() and n:
                r = f.int_in(1, n)
                if n % r == 0:
                    return mv.cast('B').cast(fmt, (r, n // r))
            return mv.cast('B').cast(fmt)
        except EXPECTED:
            return mv
    if kind == 2 and numpy is not None:  # numpy, possibly non-contiguous / transposed / 3-D
        dt = f.pick(NP_DT)
        nd = f.int_in(0, 3)
        shape = tuple(f.int_in(0, 5) for _ in range(nd))
        cnt = 1
        for s in shape:
            cnt *= s
        itemsize = numpy.dtype(dt).itemsize
        a = numpy.frombuffer(bytes(f.bytes(cnt * itemsize)), dtype=dt).reshape(shape)
        op = f.int_in(0, 4)
        if op == 1 and nd >= 2:
            a = a.T
        elif op == 2 and nd >= 1:
            a = a[::max(1, f.int_in(1, 3))]
        elif op == 3:
            a = numpy.asfortranarray(a)
        return a
    if kind == 3:                       # nested lists / tuples of numbers or matrices
        rows = f.int_in(0, 5)
        out = []
        for _ in range(rows):
            if f.bool():
                out.append([f.scalar(f.pick('idz')) for _ in range(f.int_in(0, 5))])
            else:
                out.append(matrix([f.double() for _ in range(f.int_in(0, 4))]))
        return out if f.bool() else tuple(out)
    if kind == 4:
        return f.scalar(f.pick('idz'))
    if kind == 5:
        tc = f.pick('idz')
        return matrix([f.scalar(tc) for _ in range(n)], tc=tc) if n else matrix([], tc=tc)
    if kind == 6:
        m = f.int_in(1, 5)
        return spmatrix([f.double() for _ in range(3)], [f.int_in(0, m - 1) for _ in range(3)],
                        [f.int_in(0, m - 1) for _ in range(3)], (m, m))
    return bytearray(f.bytes(n))


def TestOneInput(data):
    f = FDP(data)
    obj = guarded(make_obj, f)
    if obj is None:
        return
    args = [obj]
    kw = {}
    if f.bool():
        args.append((f.size_int(), f.size_int()) if f.bool() else f.size_int())
    if f.bool():
        kw['tc'] = f.pick(['i', 'd', 'z', 'x', 'q'])
    A = guarded(matrix, *args, **kw)
    if A is None or not small_enough(A):
        return
    # touch the result through the public interface
    guarded(lambda: (A.size, A.typecode, len(A), list(A)))
    guarded(lambda: A.T)
    guarded(lambda: matrix(A, tc=f.pick('idz')))
    guarded(lambda: memoryview(A).tolist())
    if numpy is not None:
        guarded(numpy.array, A)

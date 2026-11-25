"""Small FuzzedDataProvider for the cvxopt Python harnesses.

Every value is derived from the input bytes only; once the input is exhausted
all reads return zeros, so each input maps to one deterministic call sequence.
"""
import struct

# Errors cvxopt documents for bad arguments. SystemError is deliberately NOT
# here: it means an extension broke the C-API contract, and the driver aborts.
EXPECTED = (TypeError, ValueError, ArithmeticError, IndexError, KeyError,
            OverflowError, MemoryError, OSError, NotImplementedError,
            AttributeError, RecursionError, EOFError, BufferError,
            __import__('pickle').UnpicklingError)

MAXDIM = 12   # cap attacker-chosen sizes: big allocations only produce OOM noise


class FDP:
    def __init__(self, data):
        self.d = bytes(data)
        self.i = 0

    def remaining(self):
        return len(self.d) - self.i

    def bytes(self, n):
        b = self.d[self.i:self.i + n]
        self.i += len(b)
        return b + b'\0' * (n - len(b))

    def u8(self):
        return self.bytes(1)[0]

    def int_in(self, lo, hi):
        if hi <= lo:
            return lo
        span = hi - lo + 1
        v = int.from_bytes(self.bytes(2 if span > 256 else 1), 'little')
        return lo + v % span

    def small_int(self):
        """An int biased toward boundary values (negative, zero, past-the-end)."""
        k = self.u8()
        if k < 248:
            return (k % 40) - 20
        return struct.unpack('<i', self.bytes(4))[0]

    def size_int(self):
        """A matrix dimension: small, sometimes negative (sizes are capped so a
        huge attacker-chosen size cannot turn into an OOM instead of a bug)."""
        return self.int_in(-2, 3 * MAXDIM)

    def double(self):
        k = self.u8()
        if k < 160:
            return float((k % 21) - 10)
        if k < 240:
            return struct.unpack('<d', self.bytes(8))[0]
        return [float('inf'), float('-inf'), float('nan'), 0.5, -0.0, 1e-300, 1e300,
                2.0 ** 53, 1e-8, 3.0, -7.25, 100.0, 1e10, -1e-10, 0.1, 0.0][k - 240]

    def complex(self):
        return complex(self.double(), self.double())

    def pick(self, seq):
        return seq[self.int_in(0, len(seq) - 1)]

    def bool(self):
        return bool(self.u8() & 1)

    def dim(self, lo=0, hi=MAXDIM):
        return self.int_in(lo, hi)

    def scalar(self, tc):
        if tc == 'i':
            return self.small_int()
        if tc == 'd':
            return self.double()
        return self.complex()


def guarded(fn, *a, **kw):
    """Call fn, swallowing the documented argument errors only."""
    try:
        return fn(*a, **kw)
    except EXPECTED:
        return None


def rand_matrix(f, tc=None, m=None, n=None):
    from cvxopt import matrix
    tc = tc or f.pick('idz')
    m = f.dim(0, 8) if m is None else m
    n = f.dim(0, 8) if n is None else n
    vals = [f.scalar(tc) for _ in range(m * n)]
    return matrix(vals, (m, n), tc) if m * n else matrix([], (m, n), tc)


def rand_spmatrix(f, tc=None, m=None, n=None):
    from cvxopt import spmatrix
    tc = tc or f.pick('dz')
    m = f.dim(1, 8) if m is None else m
    n = f.dim(1, 8) if n is None else n
    nnz = f.int_in(0, 16)
    I = [f.int_in(0, m - 1) for _ in range(nnz)]
    J = [f.int_in(0, n - 1) for _ in range(nnz)]
    V = [f.scalar(tc) for _ in range(nnz)]
    return spmatrix(V, I, J, (m, n), tc)


def small_enough(A, limit=4096):
    """Dense work on A is only done when it fits a fuzzing budget."""
    try:
        return A.size[0] * A.size[1] <= limit
    except Exception:
        return False

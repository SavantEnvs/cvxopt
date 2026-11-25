"""matrix.fromfile / tofile, pickling (__reduce__ round-trip back through the
constructors), the size setter and str/repr for matrix and spmatrix
(src/C/dense.c, src/C/sparse.c, src/python/printing.py)."""
import io
import pickle
from fdp import FDP, guarded, rand_spmatrix
from cvxopt import matrix, spmatrix


def TestOneInput(data):
    f = FDP(data)
    tc = f.pick('idz')
    m, n = f.dim(0, 8), f.dim(0, 8)
    A = matrix(0, (m, n), tc)
    # the rest of the input is the raw file contents fromfile() reads
    hdr = f.u8()
    raw = f.d[f.i:]
    guarded(A.fromfile, io.BytesIO(raw))
    guarded(lambda: pickle.loads(pickle.dumps(A)))
    guarded(str, A)
    guarded(repr, A)
    out = io.BytesIO()
    guarded(A.tofile, out)
    fs = FDP(raw)
    if hdr & 1:
        guarded(setattr, A, "size", (fs.size_int(), fs.size_int()))
        guarded(str, A)
    if hdr & 2:
        S = guarded(rand_spmatrix, fs)
        if S is not None:
            guarded(lambda: pickle.loads(pickle.dumps(S)))
            guarded(str, S)
            guarded(repr, S)
            guarded(setattr, S, "size", (fs.size_int(), fs.size_int()))
            guarded(str, S)
            guarded(lambda: S.CCS)

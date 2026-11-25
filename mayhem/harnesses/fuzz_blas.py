"""cvxopt.blas.* (src/C/blas.c): level 1/2/3 BLAS wrappers with fuzzed
dimension, stride, leading-dimension and offset keywords."""
from cvxopt import blas
from linalg_common import call

M = 'M'
TABLE = [
    ('swap', 'MM'), ('scal', 'SM'), ('copy', 'MM'), ('axpy', 'MM'), ('dot', 'MM'),
    ('dotu', 'MM'), ('nrm2', 'M'), ('asum', 'M'), ('iamax', 'M'),
    ('gemv', 'MMM'), ('gbmv', 'MMM'), ('symv', 'MMM'), ('hemv', 'MMM'), ('sbmv', 'MMM'),
    ('hbmv', 'MMM'), ('trmv', 'MM'), ('tbmv', 'MM'), ('trsv', 'MM'), ('tbsv', 'MM'),
    ('ger', 'MMM'), ('geru', 'MMM'), ('syr', 'MM'), ('her', 'MM'), ('syr2', 'MMM'),
    ('her2', 'MMM'), ('gemm', 'MMM'), ('symm', 'MMM'), ('hemm', 'MMM'), ('syrk', 'MM'),
    ('herk', 'MM'), ('syr2k', 'MMM'), ('her2k', 'MMM'), ('trmm', 'MM'), ('trsm', 'MM'),
]


def TestOneInput(data):
    call(blas, TABLE, data)

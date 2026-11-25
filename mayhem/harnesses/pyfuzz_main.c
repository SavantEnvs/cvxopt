// libFuzzer driver that embeds CPython and feeds each input to
// TestOneInput(data: bytes) in the Python harness module PYFUZZ_MODULE
// (mayhem/harnesses/<module>.py). The harness drives cvxopt only through its
// public Python API; cvxopt's C extensions (src/C/*.c) are built with
// ASan+UBSan+SanCov by mayhem/build.sh and loaded from PYFUZZ_LIBDIR.
//
// Python exceptions the harness does not catch (it catches the documented
// argument errors and re-raises SystemError, which signals a C-API contract
// violation inside an extension) abort the process so they are reported.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PYFUZZ_MODULE
#error "define PYFUZZ_MODULE"
#endif

static PyObject *test_one_input;

static void die(const char *what) {
  fprintf(stderr, "pyfuzz: %s\n", what);
  if (PyErr_Occurred()) PyErr_Print();
  abort();
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc; (void)argv;
  PyStatus st;
  PyPreConfig pre;
  PyPreConfig_InitIsolatedConfig(&pre);
  // Route every Python allocation through malloc so ASan sees object bounds.
  pre.allocator = PYMEM_ALLOCATOR_MALLOC;
  st = Py_PreInitialize(&pre);
  if (PyStatus_Exception(st)) Py_ExitStatusException(st);

  PyConfig cfg;
  PyConfig_InitIsolatedConfig(&cfg);
  cfg.write_bytecode = 0;      // the image is read-only under Mayhem
  cfg.install_signal_handlers = 0;
  st = PyConfig_SetString(&cfg, &cfg.home, L"/usr");
  if (PyStatus_Exception(st)) Py_ExitStatusException(st);
  st = Py_InitializeFromConfig(&cfg);
  PyConfig_Clear(&cfg);
  if (PyStatus_Exception(st)) Py_ExitStatusException(st);

  PyObject *sys_path = PySys_GetObject("path");
  if (!sys_path) die("no sys.path");
  PyObject *p = PyUnicode_FromString(PYFUZZ_HARNESS_DIR);
  if (!p || PyList_Insert(sys_path, 0, p)) die("sys.path insert");
  Py_DECREF(p);
  p = PyUnicode_FromString(PYFUZZ_LIBDIR);
  if (!p || PyList_Insert(sys_path, 0, p)) die("sys.path insert");
  Py_DECREF(p);

  PyObject *mod = PyImport_ImportModule(PYFUZZ_MODULE);
  if (!mod) die("cannot import harness module " PYFUZZ_MODULE);
  test_one_input = PyObject_GetAttrString(mod, "TestOneInput");
  Py_DECREF(mod);
  if (!test_one_input || !PyCallable_Check(test_one_input)) die("no TestOneInput");
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (!test_one_input) LLVMFuzzerInitialize(NULL, NULL);
  PyObject *b = PyBytes_FromStringAndSize((const char *)data, (Py_ssize_t)size);
  if (!b) die("PyBytes_FromStringAndSize");
  PyObject *r = PyObject_CallOneArg(test_one_input, b);
  Py_DECREF(b);
  if (!r) die("uncaught Python exception in " PYFUZZ_MODULE ".TestOneInput");
  int rc = (r == Py_False) ? -1 : 0;   // harness returns False to reject an input
  Py_DECREF(r);
  return rc;
}

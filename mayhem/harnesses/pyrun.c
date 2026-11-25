// A plain CPython launcher (same as `python3`) built from this repo, so the
// functional oracle (mayhem/test.sh) runs the test suite through a project-built
// executable rather than the system interpreter.
#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }

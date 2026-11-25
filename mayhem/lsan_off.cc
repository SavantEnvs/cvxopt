// Leaks are not the bug class this integration fuzzes for (CPython itself keeps
// interpreter-lifetime allocations), so LeakSanitizer is turned off at build time.
extern "C" int __lsan_is_turned_off() { return 1; }

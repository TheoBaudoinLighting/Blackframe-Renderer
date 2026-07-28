#if defined(_WIN32)
#define BLACKFRAME_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define BLACKFRAME_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

BLACKFRAME_TEST_EXPORT void blackframe_unrelated_test_symbol() {}

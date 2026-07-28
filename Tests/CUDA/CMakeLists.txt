add_executable(
    BlackframeCudaSmokeTests
    CudaSmokeTest.cpp
)

target_link_libraries(
    BlackframeCudaSmokeTests
    PRIVATE
        Blackframe::CudaSharedHeaders
        Blackframe::CudaSmokeKernel
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;cuda"
)

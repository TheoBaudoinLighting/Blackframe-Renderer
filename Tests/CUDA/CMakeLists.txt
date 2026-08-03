add_executable(
    BlackframeCudaSmokeTests
    CudaSmokeTest.cpp
    TransportAbiTest.cpp
)
blackframe_set_target_role(BlackframeCudaSmokeTests test)

target_link_libraries(
    BlackframeCudaSmokeTests
    PRIVATE
        Blackframe::Cuda
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;smoke;cuda"
)

add_executable(
    BlackframeIpcTests
    LocalTransportTest.cpp
    ProtocolCodecTest.cpp
)

target_link_libraries(
    BlackframeIpcTests
    PRIVATE
        Blackframe::IPC
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeIpcTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
)

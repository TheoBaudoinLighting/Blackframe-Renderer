add_executable(
    BlackframeIpcTests
    LocalTransportTest.cpp
    ProtocolCodecTest.cpp
)
blackframe_set_target_role(BlackframeIpcTests test)

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
    PROPERTIES
        LABELS "unit;regression;nightly;cpu"
)

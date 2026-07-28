add_executable(
    BlackframeEmbreeSmokeTests
    EmbreeSmokeTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeSmokeTests test)

target_link_libraries(
    BlackframeEmbreeSmokeTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeEmbreeSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;cpu"
)

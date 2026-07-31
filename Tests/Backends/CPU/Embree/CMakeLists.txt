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
        LABELS "unit;smoke;regression;cpu"
)

add_executable(
    BlackframeEmbreeCornellRegressionTests
    CornellFrameSceneIntegrationTest.cpp
)
blackframe_set_target_role(BlackframeEmbreeCornellRegressionTests test)

target_link_libraries(
    BlackframeEmbreeCornellRegressionTests
    PRIVATE
        Blackframe::CpuEmbree
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeEmbreeCornellRegressionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;regression;cpu"
        TIMEOUT 120
)

add_executable(
    BlackframeRendererConvergenceTests
    ConvergenceMetricsTest.cpp
)
blackframe_set_target_role(BlackframeRendererConvergenceTests test)

target_link_libraries(
    BlackframeRendererConvergenceTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::Renderer
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeRendererConvergenceTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_executable(
    BlackframeRendererMisTests
    MisHeuristicsTest.cpp
)
blackframe_set_target_role(BlackframeRendererMisTests test)

target_link_libraries(
    BlackframeRendererMisTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::Renderer
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeRendererMisTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

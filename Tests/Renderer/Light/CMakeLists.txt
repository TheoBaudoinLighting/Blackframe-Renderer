add_executable(
    BlackframeRendererLightTests
    LightTest.cpp
)
blackframe_set_target_role(BlackframeRendererLightTests test)

target_link_libraries(
    BlackframeRendererLightTests
    PRIVATE
        Blackframe::Renderer
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeRendererLightTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_executable(
    BlackframeRendererTests
    CapabilityRegistryTest.cpp
    NumericPrecisionTest.cpp
    RenderConfigurationTest.cpp
    RenderSessionStateTest.cpp
)
blackframe_set_target_role(BlackframeRendererTests test)

target_link_libraries(
    BlackframeRendererTests
    PRIVATE
        Blackframe::Renderer
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeRendererTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

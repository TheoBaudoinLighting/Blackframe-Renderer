add_executable(
    BlackframeConstantTextureTests
    ConstantTextureTest.cpp
)
blackframe_set_target_role(BlackframeConstantTextureTests test)

target_link_libraries(
    BlackframeConstantTextureTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::Renderer
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeConstantTextureTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES LABELS "unit"
)

add_executable(
    BlackframeRenderTests
    RenderTest.cpp
    RenderTestsMain.cpp
)
blackframe_set_target_role(BlackframeRenderTests test)

target_link_libraries(
    BlackframeRenderTests
    PRIVATE
        Blackframe::Core
        Blackframe::ProjectOptions
        GTest::gtest
)

set_target_properties(
    BlackframeRenderTests
    PROPERTIES
        OUTPUT_NAME render_tests
        DEBUG_POSTFIX ""
)

gtest_discover_tests(
    BlackframeRenderTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_test(
    NAME Blackframe.Executables.RenderTests.Help
    COMMAND "$<TARGET_FILE:BlackframeRenderTests>" --help
)

if(TARGET BlackframeRender)
    add_test(
        NAME Blackframe.Executables.Render.Help
        COMMAND "$<TARGET_FILE:BlackframeRender>" --help
    )
    set_tests_properties(
        Blackframe.Executables.Render.Help
        PROPERTIES
            LABELS "smoke;cpu"
            PASS_REGULAR_EXPRESSION "render --help"
    )
endif()

set_tests_properties(
    Blackframe.Executables.RenderTests.Help
    PROPERTIES
        LABELS "smoke;cpu"
        PASS_REGULAR_EXPRESSION "render_tests --help"
)

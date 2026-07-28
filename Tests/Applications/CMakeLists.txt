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

    if(BLACKFRAME_BUILD_REFERENCE_PLUGINS)
        set(reference_cpu_status experimental)
    else()
        set(reference_cpu_status unavailable)
    endif()
    if(BLACKFRAME_ENABLE_EMBREE)
        set(embree_status supported)
    else()
        set(embree_status unavailable)
    endif()
    if(BLACKFRAME_ENABLE_CUDA)
        set(cuda_status supported)
    else()
        set(cuda_status unavailable)
    endif()

    add_test(
        NAME Blackframe.Executables.Render.Capabilities
        COMMAND
            "${CMAKE_COMMAND}"
            "-DRENDER_EXECUTABLE=$<TARGET_FILE:BlackframeRender>"
            "-DREFERENCE_CPU_STATUS=${reference_cpu_status}"
            "-DEMBREE_STATUS=${embree_status}"
            "-DCUDA_STATUS=${cuda_status}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/ValidateCapabilityManifest.cmake"
    )
    set_tests_properties(
        Blackframe.Executables.Render.Capabilities
        PROPERTIES
            LABELS "smoke;cpu"
    )
endif()

set_tests_properties(
    Blackframe.Executables.RenderTests.Help
    PROPERTIES
        LABELS "smoke;cpu"
        PASS_REGULAR_EXPRESSION "render_tests --help"
)

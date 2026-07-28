add_executable(
    BlackframeXpuTests
    XpuBackendTest.cpp
)
blackframe_set_target_role(BlackframeXpuTests test)

target_link_libraries(
    BlackframeXpuTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::XpuHost
        GTest::gtest_main
)

if(TARGET BlackframeXpuReferenceCpu)
    add_dependencies(BlackframeXpuTests BlackframeXpuReferenceCpu)
    target_compile_definitions(
        BlackframeXpuTests
        PRIVATE
            BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH="$<TARGET_FILE:BlackframeXpuReferenceCpu>"
    )
else()
    message(
        FATAL_ERROR
        "XPU integration tests require BLACKFRAME_BUILD_REFERENCE_PLUGINS=ON."
    )
endif()

gtest_discover_tests(
    BlackframeXpuTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
)

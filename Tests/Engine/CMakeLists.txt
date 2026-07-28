add_executable(
    BlackframeEngineTests
    EngineTest.cpp
)

target_link_libraries(
    BlackframeEngineTests
    PRIVATE
        Blackframe::Engine
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET BlackframeXpuReferenceCpu)
    add_dependencies(BlackframeEngineTests BlackframeXpuReferenceCpu)
    target_compile_definitions(
        BlackframeEngineTests
        PRIVATE
            BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH="$<TARGET_FILE:BlackframeXpuReferenceCpu>"
    )
else()
    message(FATAL_ERROR "Engine tests require BLACKFRAME_BUILD_REFERENCE_PLUGINS=ON.")
endif()

gtest_discover_tests(
    BlackframeEngineTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
)

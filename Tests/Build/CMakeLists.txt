add_executable(
    BlackframeHostLanguageTests
    Cxx26LanguageFeatureTest.cpp
)

target_link_libraries(
    BlackframeHostLanguageTests
    PRIVATE
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeHostLanguageTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_test(
    NAME Blackframe.Build.HostLanguage.RejectsCompilerWithoutCxx26Capability
    COMMAND
        "${CMAKE_COMMAND}"
        "-DBLACKFRAME_HOST_LANGUAGE_MODULE=${CMAKE_SOURCE_DIR}/cmake/BlackframeHostLanguage.cmake"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/IncompatibleCompilerRejectionTest.cmake"
)

set_tests_properties(
    Blackframe.Build.HostLanguage.RejectsCompilerWithoutCxx26Capability
    PROPERTIES
        LABELS "unit;cpu"
)

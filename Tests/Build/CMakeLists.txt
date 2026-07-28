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

add_test(
    NAME Blackframe.Build.DependencyManifest
    COMMAND
        "${CMAKE_COMMAND}"
        "-DMANIFEST_PATH=${BLACKFRAME_DEPENDENCY_MANIFEST}"
        "-DMANIFEST_SHA256=${BLACKFRAME_DEPENDENCY_MANIFEST_SHA256}"
        "-DGOOGLETEST_VERSION=${BLACKFRAME_GOOGLETEST_VERSION}"
        "-DGOOGLETEST_REVISION=${BLACKFRAME_GOOGLETEST_REVISION}"
        "-DGOOGLETEST_SHA256=${BLACKFRAME_GOOGLETEST_SHA256}"
        "-DGOOGLE_BENCHMARK_VERSION=${BLACKFRAME_GOOGLE_BENCHMARK_VERSION}"
        "-DGOOGLE_BENCHMARK_REVISION=${BLACKFRAME_GOOGLE_BENCHMARK_REVISION}"
        "-DGOOGLE_BENCHMARK_SHA256=${BLACKFRAME_GOOGLE_BENCHMARK_SHA256}"
        "-DEMBREE_VERSION=${BLACKFRAME_EMBREE_VERSION}"
        "-DEMBREE_REVISION=${BLACKFRAME_EMBREE_REVISION}"
        "-DEMBREE_SHA256=${BLACKFRAME_EMBREE_SHA256}"
        "-DSTB_REVISION=${BLACKFRAME_STB_REVISION}"
        "-DSTB_SHA256=${BLACKFRAME_STB_SHA256}"
        "-DCUDA_TOOLKIT_VERSION=${BLACKFRAME_CUDA_TOOLKIT_VERSION}"
        "-DGOOGLE_BENCHMARK_ENABLED=${BLACKFRAME_BUILD_BENCHMARKS}"
        "-DEMBREE_ENABLED=${BLACKFRAME_ENABLE_EMBREE}"
        "-DSTB_ENABLED=${BLACKFRAME_ENABLE_STB}"
        "-DCUDA_ENABLED=${BLACKFRAME_ENABLE_CUDA}"
        "-DCUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/DependencyManifestTest.cmake"
)

set_tests_properties(
    Blackframe.Build.DependencyManifest
    PROPERTIES
        LABELS "unit;cpu"
)

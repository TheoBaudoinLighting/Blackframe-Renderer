add_executable(
    BlackframeHostLanguageTests
    Cxx26LanguageFeatureTest.cpp
)
blackframe_set_target_role(BlackframeHostLanguageTests test)

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

if(NOT BLACKFRAME_ENABLE_CUDA)
    add_test(
        NAME Blackframe.Capabilities.CudaUnavailable
        COMMAND
            "${CMAKE_COMMAND}"
            "-DMANIFEST_PATH=${BLACKFRAME_DEPENDENCY_MANIFEST}"
            "-DTARGET_GRAPH_PATH=${CMAKE_BINARY_DIR}/blackframe-target-graph.json"
            -P
            "${CMAKE_CURRENT_SOURCE_DIR}/CudaCapabilityUnavailableTest.cmake"
    )
    set_tests_properties(
        Blackframe.Capabilities.CudaUnavailable
        PROPERTIES
            LABELS "smoke;cpu"
            PASS_REGULAR_EXPRESSION
                "CUDA capability is explicitly unavailable in this CPU-only build"
    )
endif()

add_test(
    NAME Blackframe.Build.TargetGraph
    COMMAND
        "${CMAKE_COMMAND}"
        "-DTARGET_GRAPH_PATH=${CMAKE_BINARY_DIR}/blackframe-target-graph.json"
        "-DCPU_EMBREE_ENABLED=${BLACKFRAME_ENABLE_EMBREE}"
        "-DCUDA_ENABLED=${BLACKFRAME_ENABLE_CUDA}"
        "-DTOOLS_ENABLED=${BLACKFRAME_BUILD_TOOLS}"
        "-DBENCHMARKS_ENABLED=${BLACKFRAME_BUILD_BENCHMARKS}"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/TargetGraphTest.cmake"
)

add_test(
    NAME Blackframe.Build.TargetGraph.RejectsCycle
    COMMAND
        "${CMAKE_COMMAND}"
        "-DPROBE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/TargetGraphCycle"
        "-DPROBE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/TargetGraphCycle"
        "-DTARGET_GRAPH_MODULE=${CMAKE_SOURCE_DIR}/cmake/BlackframeTargetGraph.cmake"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/TargetGraphCycleRejectionTest.cmake"
)

set_tests_properties(
    Blackframe.Build.TargetGraph
    Blackframe.Build.TargetGraph.RejectsCycle
    PROPERTIES
        LABELS "unit;cpu"
)

set(blackframe_required_ctest_labels "unit,smoke,regression,nightly,cpu")
if(BLACKFRAME_ENABLE_CUDA)
    string(APPEND blackframe_required_ctest_labels ",cuda")
endif()

add_test(
    NAME Blackframe.Build.CTestLabelInventory
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        "-DBINARY_DIRECTORY=${CMAKE_BINARY_DIR}"
        "-DREQUIRED_LABELS=${blackframe_required_ctest_labels}"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/CTestLabelInventoryTest.cmake"
)
set_tests_properties(
    Blackframe.Build.CTestLabelInventory
    PROPERTIES
        LABELS "smoke;cpu"
)

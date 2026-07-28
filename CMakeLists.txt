cmake_minimum_required(VERSION 3.30)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(BlackframeBuildTree)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    blackframe_require_disposable_build_tree()
endif()

project(
    Blackframe
    VERSION 0.1.0
    DESCRIPTION "A headless, extensible XPU rendering engine"
    LANGUAGES C CXX
)

include(CTest)
include(GNUInstallDirs)
include(BlackframeDependencies)
include(BlackframeCudaLanguage)
include(BlackframeHostLanguage)
include(BlackframeProjectOptions)

option(BLACKFRAME_BUILD_BENCHMARKS "Build Blackframe microbenchmarks" ${PROJECT_IS_TOP_LEVEL})
option(BLACKFRAME_BUILD_REFERENCE_PLUGINS "Build reference XPU plugins" ${PROJECT_IS_TOP_LEVEL})
option(BLACKFRAME_BUILD_DCC_MAYA "Build the Maya DCC plugin (requires the Maya SDK)" OFF)
option(BLACKFRAME_ENABLE_EMBREE "Fetch the Embree 4 CPU ray-query dependency" ${PROJECT_IS_TOP_LEVEL})
option(BLACKFRAME_ENABLE_STB "Fetch the stb image utility headers" ${PROJECT_IS_TOP_LEVEL})
option(BLACKFRAME_ENABLE_CUDA "Expose the locally installed CUDA toolkit" ${PROJECT_IS_TOP_LEVEL})
option(BLACKFRAME_WARNINGS_AS_ERRORS "Treat warnings in Blackframe targets as errors" OFF)
option(BLACKFRAME_ENABLE_SANITIZERS "Enable supported address/undefined sanitizers" OFF)

set(blackframe_plugin_output_directory "${CMAKE_BINARY_DIR}/plugins")

if(BLACKFRAME_ENABLE_EMBREE)
    blackframe_fetch_embree()
endif()

if(BLACKFRAME_ENABLE_STB)
    blackframe_fetch_stb()
endif()

if(BUILD_TESTING)
    blackframe_fetch_googletest()
endif()

if(BLACKFRAME_BUILD_BENCHMARKS)
    blackframe_fetch_google_benchmark()
endif()

if(BLACKFRAME_ENABLE_CUDA)
    blackframe_require_explicit_cuda_architectures()
endif()

blackframe_require_host_cxx26()
blackframe_create_project_options()

add_subdirectory(Blackframe)

blackframe_write_dependency_manifest(
    "${CMAKE_BINARY_DIR}/blackframe-dependencies.json"
)

if(BLACKFRAME_BUILD_REFERENCE_PLUGINS)
    add_subdirectory(Plugins/XPU/ReferenceCpu)
endif()

add_subdirectory(DCC)

if(BUILD_TESTING)
    add_subdirectory(Tests)
endif()

if(BLACKFRAME_BUILD_BENCHMARKS)
    add_subdirectory(Benchmarks)
endif()

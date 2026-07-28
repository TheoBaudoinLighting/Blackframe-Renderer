include_guard(GLOBAL)

include(FetchContent)

set(
    FETCHCONTENT_BASE_DIR
    "${CMAKE_BINARY_DIR}/_deps"
    CACHE PATH
    "Location of dependencies populated by FetchContent"
    FORCE
)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update populated dependencies" FORCE)

function(blackframe_fetch_googletest)
    if(TARGET GTest::gtest_main)
        return()
    endif()

    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        URL
            https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.tar.gz
        URL_HASH
            SHA256=745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
endfunction()

function(blackframe_fetch_google_benchmark)
    if(TARGET benchmark::benchmark)
        return()
    endif()

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googlebenchmark
        URL
            https://github.com/google/benchmark/archive/192ef10025eb2c4cdd392bc502f0c852196baa48.tar.gz
        URL_HASH
            SHA256=f82705a2726d8f6cdcda274b841f6314dbfc6f731cdda06c946f310ec1cc3ad9
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googlebenchmark)
endfunction()

function(blackframe_fetch_embree)
    if(TARGET Blackframe::Embree)
        return()
    endif()

    set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
    set(EMBREE_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
    set(EMBREE_SYCL_SUPPORT OFF CACHE BOOL "" FORCE)
    set(EMBREE_TASKING_SYSTEM INTERNAL CACHE STRING "" FORCE)
    set(EMBREE_INSTALL_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(EMBREE_TESTING_INTENSITY 0 CACHE STRING "" FORCE)

    # Embree's test CMake defines helper macros only when BUILD_TESTING is true,
    # even at intensity zero. Keep that local requirement from changing the
    # Blackframe option seen after this function returns.
    set(BUILD_TESTING ON)

    FetchContent_Declare(
        embree
        URL
            https://github.com/RenderKit/embree/archive/f590db83ef6559387df7f6d8725c34fb7acf851d.tar.gz
        URL_HASH
            SHA256=7301c78b4fc6d3ea302601057d8dbe5b97ae30fd3fcc2c1be82d05555bf108e6
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(embree)

    if(NOT TARGET embree)
        message(FATAL_ERROR "The pinned Embree source did not provide the expected `embree` target.")
    endif()

    add_library(BlackframeEmbree INTERFACE)
    add_library(Blackframe::Embree ALIAS BlackframeEmbree)
    target_link_libraries(BlackframeEmbree INTERFACE embree)
endfunction()

function(blackframe_fetch_stb)
    if(TARGET Blackframe::Stb)
        return()
    endif()

    FetchContent_Declare(
        stb
        URL
            https://github.com/nothings/stb/archive/31c1ad37456438565541f4919958214b6e762fb4.tar.gz
        URL_HASH
            SHA256=e4e3bba9c572a4a4148373a914d88ea0f0d11de8cc2c66739926e7eca0223319
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(stb)

    add_library(BlackframeStb INTERFACE)
    add_library(Blackframe::Stb ALIAS BlackframeStb)
    target_include_directories(BlackframeStb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
endfunction()

function(blackframe_find_cuda_toolkit)
    if(TARGET Blackframe::CudaToolkit)
        return()
    endif()

    find_package(CUDAToolkit REQUIRED)

    add_library(BlackframeCudaToolkit INTERFACE)
    add_library(Blackframe::CudaToolkit ALIAS BlackframeCudaToolkit)
    target_link_libraries(
        BlackframeCudaToolkit
        INTERFACE
            CUDA::cuda_driver
            CUDA::cudart
    )
endfunction()

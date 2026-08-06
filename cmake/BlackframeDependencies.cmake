include_guard(GLOBAL)

include(FetchContent)

set(BLACKFRAME_GOOGLETEST_VERSION "1.17.0")
set(BLACKFRAME_GOOGLETEST_REVISION "52eb8108c5bdec04579160ae17225d66034bd723")
set(BLACKFRAME_GOOGLETEST_SHA256
    "745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be"
)

set(BLACKFRAME_GOOGLE_BENCHMARK_VERSION "1.9.5")
set(BLACKFRAME_GOOGLE_BENCHMARK_REVISION "192ef10025eb2c4cdd392bc502f0c852196baa48")
set(BLACKFRAME_GOOGLE_BENCHMARK_SHA256
    "f82705a2726d8f6cdcda274b841f6314dbfc6f731cdda06c946f310ec1cc3ad9"
)

set(BLACKFRAME_EMBREE_VERSION "4.4.1")
set(BLACKFRAME_EMBREE_REVISION "f590db83ef6559387df7f6d8725c34fb7acf851d")
set(BLACKFRAME_EMBREE_SHA256
    "7301c78b4fc6d3ea302601057d8dbe5b97ae30fd3fcc2c1be82d05555bf108e6"
)

set(BLACKFRAME_STB_REVISION "31c1ad37456438565541f4919958214b6e762fb4")
set(BLACKFRAME_STB_SHA256
    "e4e3bba9c572a4a4148373a914d88ea0f0d11de8cc2c66739926e7eca0223319"
)

set(BLACKFRAME_IMATH_VERSION "3.2.2")
set(BLACKFRAME_IMATH_REVISION "1e480d11cb98b032a2dece9b9a8730512effc7f6")
set(BLACKFRAME_IMATH_SHA256
    "e5847ee5f19aa6adfe5512fd05584338e7656cd84c6f7644707251c6f3fa0cdb"
)

set(BLACKFRAME_OPENEXR_VERSION "3.4.13")
set(BLACKFRAME_OPENEXR_REVISION "c1194b2cb23a1bdf76fe5e756b22e8436b9a98c9")
set(BLACKFRAME_OPENEXR_SHA256
    "c4a5fb903facf83a1bffcce25a8fed931bfa4d179b3aa8d0541069f56644d7aa"
)

set(BLACKFRAME_CUDA_TOOLKIT_VERSION "13.3.33")

set(
    BLACKFRAME_GOOGLETEST_URL
    "https://github.com/google/googletest/archive/${BLACKFRAME_GOOGLETEST_REVISION}.tar.gz"
)
set(
    BLACKFRAME_GOOGLE_BENCHMARK_URL
    "https://github.com/google/benchmark/archive/${BLACKFRAME_GOOGLE_BENCHMARK_REVISION}.tar.gz"
)
set(
    BLACKFRAME_EMBREE_URL
    "https://github.com/RenderKit/embree/archive/${BLACKFRAME_EMBREE_REVISION}.tar.gz"
)
set(
    BLACKFRAME_STB_URL
    "https://github.com/nothings/stb/archive/${BLACKFRAME_STB_REVISION}.tar.gz"
)
set(
    BLACKFRAME_IMATH_URL
    "https://github.com/AcademySoftwareFoundation/Imath/archive/${BLACKFRAME_IMATH_REVISION}.tar.gz"
)
set(
    BLACKFRAME_OPENEXR_URL
    "https://github.com/AcademySoftwareFoundation/openexr/archive/${BLACKFRAME_OPENEXR_REVISION}.tar.gz"
)

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
            "${BLACKFRAME_GOOGLETEST_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_GOOGLETEST_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)

    if(NOT TARGET GTest::gtest_main)
        message(
            FATAL_ERROR
            "GoogleTest ${BLACKFRAME_GOOGLETEST_VERSION} did not provide GTest::gtest_main."
        )
    endif()
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
            "${BLACKFRAME_GOOGLE_BENCHMARK_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_GOOGLE_BENCHMARK_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googlebenchmark)

    if(NOT TARGET benchmark::benchmark)
        message(
            FATAL_ERROR
            "Google Benchmark ${BLACKFRAME_GOOGLE_BENCHMARK_VERSION} did not provide "
            "benchmark::benchmark."
        )
    endif()
endfunction()

function(blackframe_fetch_embree)
    if(TARGET Blackframe::Embree)
        return()
    endif()

    set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
    set(EMBREE_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
    set(EMBREE_SYCL_SUPPORT OFF CACHE BOOL "" FORCE)
    set(EMBREE_GEOMETRY_INSTANCE ON CACHE BOOL "" FORCE)
    set(EMBREE_MAX_INSTANCE_LEVEL_COUNT 1 CACHE STRING "" FORCE)
    set(EMBREE_RAY_MASK ON CACHE BOOL "" FORCE)
    set(EMBREE_BACKFACE_CULLING OFF CACHE BOOL "" FORCE)
    set(EMBREE_TASKING_SYSTEM INTERNAL CACHE STRING "" FORCE)
    set(EMBREE_INSTALL_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(EMBREE_TESTING_INTENSITY 0 CACHE STRING "" FORCE)

    # Embree's Clang flags still request C++11, which is too old for the
    # current MSVC standard library used by clang++. A later CMake-managed
    # standard flag keeps the dependency build portable without leaking into
    # Blackframe targets.
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Embree's test CMake defines helper macros only when BUILD_TESTING is true,
    # even at intensity zero. Keep that local requirement from changing the
    # Blackframe option seen after this function returns.
    set(BUILD_TESTING ON)

    FetchContent_Declare(
        embree
        URL
            "${BLACKFRAME_EMBREE_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_EMBREE_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(embree)

    if(NOT TARGET embree)
        message(FATAL_ERROR "The pinned Embree source did not provide the expected `embree` target.")
    endif()

    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # A static Embree archive does not need Windows version metadata.
        # Removing the resource avoids mixing the SDK resource compiler with
        # Clang dependency flags when nvcc selects an MSVC host environment.
        get_target_property(blackframe_embree_sources embree SOURCES)
        list(FILTER blackframe_embree_sources EXCLUDE REGEX "\\.rc$")
        set_property(TARGET embree PROPERTY SOURCES "${blackframe_embree_sources}")
    endif()

    add_library(BlackframeEmbree INTERFACE)
    add_library(Blackframe::Embree ALIAS BlackframeEmbree)
    target_link_libraries(BlackframeEmbree INTERFACE embree)
    blackframe_set_target_role(BlackframeEmbree dependency)
endfunction()

function(blackframe_fetch_stb)
    if(TARGET Blackframe::Stb)
        return()
    endif()

    FetchContent_Declare(
        stb
        URL
            "${BLACKFRAME_STB_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_STB_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(stb)

    add_library(BlackframeStb INTERFACE)
    add_library(Blackframe::Stb ALIAS BlackframeStb)
    target_include_directories(BlackframeStb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
    blackframe_set_target_role(BlackframeStb dependency)
endfunction()

function(blackframe_fetch_openexr)
    if(TARGET Blackframe::OpenExr)
        return()
    endif()

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL OFF CACHE BOOL "" FORCE)
    set(IMATH_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(IMATH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PYTHON OFF CACHE BOOL "" FORCE)
    set(PYBIND11 OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        imath
        URL
            "${BLACKFRAME_IMATH_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_IMATH_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(imath)
    if(NOT TARGET Imath::Imath)
        message(
            FATAL_ERROR
            "Imath ${BLACKFRAME_IMATH_VERSION} did not provide Imath::Imath."
        )
    endif()

    set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_DEVELOPER_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_LIBS ON CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(OPENEXR_BUILD_OSS_FUZZ OFF CACHE BOOL "" FORCE)
    set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL "" FORCE)
    set(OPENEXR_FORCE_INTERNAL_OPENJPH ON CACHE BOOL "" FORCE)
    # The pinned Imath target is already present. Prevent OpenEXR from probing a
    # system package or re-fetching its tag-based fallback.
    set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        openexr
        URL
            "${BLACKFRAME_OPENEXR_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_OPENEXR_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(openexr)
    if(NOT TARGET OpenEXR::OpenEXR)
        message(
            FATAL_ERROR
            "OpenEXR ${BLACKFRAME_OPENEXR_VERSION} did not provide OpenEXR::OpenEXR."
        )
    endif()
    if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang" AND
       CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM")
        # OpenEXR's Windows ZIP path follows MSVC and enables its SSE4.1
        # intrinsics on x64. The GNU-style Clang driver additionally requires
        # the matching target feature to compile those same intrinsics.
        target_compile_options(OpenEXRCore PRIVATE -msse4.1)
        target_compile_options(OpenEXR PRIVATE -msse4.1)
    endif()

    add_library(BlackframeOpenExr INTERFACE)
    add_library(Blackframe::OpenExr ALIAS BlackframeOpenExr)
    target_link_libraries(
        BlackframeOpenExr
        INTERFACE
            OpenEXR::OpenEXR
    )
    blackframe_set_target_role(BlackframeOpenExr dependency)
endfunction()

function(blackframe_find_cuda_toolkit)
    if(TARGET Blackframe::CudaToolkit)
        return()
    endif()

    find_package(CUDAToolkit "${BLACKFRAME_CUDA_TOOLKIT_VERSION}" EXACT REQUIRED)

    if(NOT CUDAToolkit_VERSION VERSION_EQUAL BLACKFRAME_CUDA_TOOLKIT_VERSION)
        message(
            FATAL_ERROR
            "CUDA Toolkit ${BLACKFRAME_CUDA_TOOLKIT_VERSION} is required exactly; "
            "found ${CUDAToolkit_VERSION}."
        )
    endif()

    foreach(cuda_target IN ITEMS CUDA::cuda_driver CUDA::cudart CUDA::nvtx3)
        if(NOT TARGET "${cuda_target}")
            message(
                FATAL_ERROR
                "CUDA Toolkit ${BLACKFRAME_CUDA_TOOLKIT_VERSION} did not provide ${cuda_target}."
            )
        endif()
    endforeach()

    add_library(BlackframeCudaToolkit INTERFACE)
    add_library(Blackframe::CudaToolkit ALIAS BlackframeCudaToolkit)
    target_link_libraries(
        BlackframeCudaToolkit
        INTERFACE
            CUDA::cuda_driver
            CUDA::cudart
    )
    blackframe_set_target_role(BlackframeCudaToolkit dependency)

    add_library(BlackframeCudaNvtx3 INTERFACE)
    add_library(Blackframe::CudaNvtx3 ALIAS BlackframeCudaNvtx3)
    target_link_libraries(BlackframeCudaNvtx3 INTERFACE CUDA::nvtx3)
    blackframe_set_target_role(BlackframeCudaNvtx3 dependency)

    set_property(
        GLOBAL
        PROPERTY BLACKFRAME_CUDA_TOOLKIT_RESOLVED_VERSION "${CUDAToolkit_VERSION}"
    )
endfunction()

function(blackframe_write_dependency_manifest output_path)
    cmake_path(ABSOLUTE_PATH output_path NORMALIZE OUTPUT_VARIABLE manifest_path)
    cmake_path(
        IS_PREFIX
        CMAKE_BINARY_DIR
        "${manifest_path}"
        NORMALIZE
        manifest_is_in_build_tree
    )
    if(NOT manifest_is_in_build_tree)
        message(
            FATAL_ERROR
            "The dependency manifest must be generated below '${CMAKE_BINARY_DIR}'."
        )
    endif()

    if(BUILD_TESTING)
        set(manifest_googletest_enabled true)
    else()
        set(manifest_googletest_enabled false)
    endif()

    if(BLACKFRAME_BUILD_BENCHMARKS)
        set(manifest_google_benchmark_enabled true)
    else()
        set(manifest_google_benchmark_enabled false)
    endif()

    if(BLACKFRAME_ENABLE_EMBREE)
        set(manifest_embree_enabled true)
    else()
        set(manifest_embree_enabled false)
    endif()

    if(BLACKFRAME_ENABLE_STB)
        set(manifest_stb_enabled true)
    else()
        set(manifest_stb_enabled false)
    endif()

    if(BLACKFRAME_ENABLE_CUDA)
        set(manifest_cuda_enabled true)
        get_property(
            manifest_cuda_resolved_version
            GLOBAL
            PROPERTY BLACKFRAME_CUDA_TOOLKIT_RESOLVED_VERSION
        )
        if(NOT manifest_cuda_resolved_version)
            message(FATAL_ERROR "CUDA is enabled but its resolved version was not recorded.")
        endif()
        set(manifest_cuda_resolved_version_json "\"${manifest_cuda_resolved_version}\"")
    else()
        set(manifest_cuda_enabled false)
        set(manifest_cuda_resolved_version_json null)
    endif()

    set(manifest_cuda_architectures_json "")
    if(BLACKFRAME_ENABLE_CUDA)
        foreach(cuda_architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
            if(manifest_cuda_architectures_json)
                string(APPEND manifest_cuda_architectures_json ", ")
            endif()
            string(APPEND manifest_cuda_architectures_json "\"${cuda_architecture}\"")
        endforeach()
    endif()

    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BlackframeDependencyManifest.json.in"
        "${manifest_path}"
        @ONLY
        NEWLINE_STYLE UNIX
    )

    file(SHA256 "${manifest_path}" manifest_sha256)
    get_filename_component(manifest_name "${manifest_path}" NAME)
    set(manifest_checksum_path "${manifest_path}.sha256")
    file(WRITE "${manifest_checksum_path}" "${manifest_sha256}  ${manifest_name}\n")

    set(BLACKFRAME_DEPENDENCY_MANIFEST "${manifest_path}" CACHE INTERNAL "" FORCE)
    set(
        BLACKFRAME_DEPENDENCY_MANIFEST_SHA256
        "${manifest_sha256}"
        CACHE INTERNAL
        ""
        FORCE
    )

    message(STATUS "Blackframe dependency manifest: ${manifest_path}")
endfunction()

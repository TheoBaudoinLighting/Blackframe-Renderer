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

set(BLACKFRAME_OPENIMAGEIO_VERSION "3.1.16.0")
set(BLACKFRAME_OPENIMAGEIO_REVISION "f3f12ab761fbccaa644937780e7246674033c407")
set(BLACKFRAME_OPENIMAGEIO_SHA256
    "0ebe0a55d8e6aeac5a039c203d8aa4de0c1abef7a134d16c71bb2e4f68f047b6"
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
    BLACKFRAME_OPENIMAGEIO_URL
    "https://github.com/AcademySoftwareFoundation/OpenImageIO/archive/${BLACKFRAME_OPENIMAGEIO_REVISION}.tar.gz"
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
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(imath)
    if(NOT TARGET Imath::Imath)
        message(
            FATAL_ERROR
            "Imath ${BLACKFRAME_IMATH_VERSION} did not provide Imath::Imath."
        )
    endif()
    # The build-tree target exposes Imath's historical flat include root.
    # OIIO also uses the installed-style <Imath/...> spelling.
    target_include_directories(
        Imath
        SYSTEM
        INTERFACE
            "$<BUILD_INTERFACE:${imath_SOURCE_DIR}/src>"
    )

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
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(openexr)
    if(NOT TARGET OpenEXR::OpenEXR)
        message(
            FATAL_ERROR
            "OpenEXR ${BLACKFRAME_OPENEXR_VERSION} did not provide OpenEXR::OpenEXR."
        )
    endif()

    # Match the installed <OpenEXR/...> include layout while consuming the
    # pinned library directly from its build tree. OpenEXR keeps Iex,
    # IlmThread, OpenEXR, and OpenEXRCore headers in separate source folders.
    set(openexr_staged_include_root "${CMAKE_BINARY_DIR}/dependency-includes")
    set(openexr_staged_include_dir "${openexr_staged_include_root}/OpenEXR")
    file(MAKE_DIRECTORY "${openexr_staged_include_dir}")
    foreach(openexr_header_directory IN ITEMS Iex IlmThread OpenEXR OpenEXRCore)
        file(
            GLOB
            openexr_public_headers
            CONFIGURE_DEPENDS
            "${openexr_SOURCE_DIR}/src/lib/${openexr_header_directory}/*.h"
        )
        file(COPY ${openexr_public_headers} DESTINATION "${openexr_staged_include_dir}")
    endforeach()
    file(
        GLOB
        openexr_generated_headers
        CONFIGURE_DEPENDS
        "${openexr_BINARY_DIR}/cmake/*.h"
    )
    file(COPY ${openexr_generated_headers} DESTINATION "${openexr_staged_include_dir}")

    target_include_directories(
        OpenEXR
        SYSTEM
        INTERFACE
            "$<BUILD_INTERFACE:${openexr_SOURCE_DIR}/src/lib>"
            "$<BUILD_INTERFACE:${openexr_staged_include_root}>"
    )
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

function(blackframe_fetch_openimageio)
    if(TARGET Blackframe::OpenImageIo)
        return()
    endif()
    if(NOT TARGET OpenEXR::OpenEXR OR NOT TARGET Imath::Imath)
        message(FATAL_ERROR "OpenImageIO requires the pinned OpenEXR and Imath targets.")
    endif()

    # Keep the dependency static without changing the parent project's cache.
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)
    set(OIIO_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(OIIO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF)
    set(INSTALL_DOCS OFF)
    set(INSTALL_FONTS OFF)
    set(USE_PYTHON OFF)
    set(USE_QT OFF)
    set(EMBEDPLUGINS ON CACHE BOOL "" FORCE)
    set(OIIO_BUILD_FUZZ_TARGETS OFF CACHE BOOL "" FORCE)
    set(OIIO_INTERNALIZE_FMT ON CACHE BOOL "" FORCE)
    set(ENABLE_PNG ON)
    set(ENABLE_OPENEXR ON)
    set(ENABLE_PNM ON)

    if(CMAKE_BUILD_TYPE)
        set(openimageio_dependency_build_type "${CMAKE_BUILD_TYPE}")
    elseif("Release" IN_LIST CMAKE_CONFIGURATION_TYPES)
        set(openimageio_dependency_build_type Release)
    elseif(CMAKE_CONFIGURATION_TYPES)
        list(LENGTH CMAKE_CONFIGURATION_TYPES openimageio_configuration_count)
        if(NOT openimageio_configuration_count EQUAL 1)
            message(
                FATAL_ERROR
                "OpenImageIO's configure-time dependency build requires a "
                "single-configuration build tree. Use a provided Ninja preset "
                "instead of selecting a configuration after configure time."
            )
        endif()
        list(GET CMAKE_CONFIGURATION_TYPES 0 openimageio_dependency_build_type)
    else()
        message(FATAL_ERROR "OpenImageIO requires an explicit dependency build configuration.")
    endif()
    if(NOT openimageio_dependency_build_type)
        message(FATAL_ERROR "OpenImageIO requires an explicit dependency build configuration.")
    endif()
    set(
        OpenImageIO_DEPENDENCY_BUILD_TYPE
        "${openimageio_dependency_build_type}"
        CACHE STRING
        ""
        FORCE
    )

    # OIIO treats ZLIB, JPEG, TIFF, OpenColorIO, Robinmap, and fmt as required.
    # Its local dependency recipes pin and verify every source commit. PNG is
    # promoted to the same fail-fast contract because Blackframe validates it.
    set(OpenImageIO_REQUIRED_DEPS "PNG" CACHE STRING "" FORCE)
    set(
        OpenImageIO_BUILD_MISSING_DEPS
        "required;libjpeg-turbo;pystring"
        CACHE STRING
        ""
        FORCE
    )
    file(
        SHA256
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BlackframePatchOpenImageIo.cmake"
        openimageio_patch_sha256
    )
    file(
        SHA256
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BlackframeOpenImageIoToolchain.cmake.in"
        openimageio_toolchain_template_sha256
    )
    file(
        SHA256
        "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        openimageio_dependency_logic_sha256
    )
    set(openimageio_parent_toolchain_sha256 "none")
    if(CMAKE_TOOLCHAIN_FILE)
        if(NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
            message(FATAL_ERROR "The selected parent toolchain file does not exist.")
        endif()
        file(
            SHA256
            "${CMAKE_TOOLCHAIN_FILE}"
            openimageio_parent_toolchain_sha256
        )
    endif()
    string(
        SHA256
        openimageio_local_dependency_key
        "${BLACKFRAME_OPENIMAGEIO_REVISION};${openimageio_patch_sha256};${openimageio_toolchain_template_sha256};${openimageio_dependency_logic_sha256};${openimageio_parent_toolchain_sha256};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_SIZEOF_VOID_P};${CMAKE_GENERATOR};${CMAKE_GENERATOR_INSTANCE};${CMAKE_GENERATOR_PLATFORM};${CMAKE_GENERATOR_TOOLSET};${CMAKE_MAKE_PROGRAM};${CMAKE_C_COMPILER};${CMAKE_C_COMPILER_ID};${CMAKE_C_COMPILER_VERSION};${CMAKE_C_COMPILER_TARGET};${CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN};${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_CXX_COMPILER_FRONTEND_VARIANT};${CMAKE_CXX_COMPILER_TARGET};${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN};${CMAKE_SYSROOT};${CMAKE_OSX_ARCHITECTURES};${CMAKE_OSX_DEPLOYMENT_TARGET};${CMAKE_TOOLCHAIN_FILE};${openimageio_dependency_build_type};${CMAKE_MSVC_RUNTIME_LIBRARY}"
    )
    string(
        SUBSTRING
        "${openimageio_local_dependency_key}"
        0
        20
        openimageio_local_dependency_key_short
    )
    set(
        openimageio_local_dependency_root
        "${FETCHCONTENT_BASE_DIR}/openimageio-build/deps-${openimageio_local_dependency_key_short}"
    )

    # Package find modules cache concrete headers and archives. When the
    # dependency/toolchain key changes, discard only entries that point into a
    # previous private OIIO prefix so two closures can never be mixed.
    if(NOT DEFINED BLACKFRAME_OPENIMAGEIO_DEPENDENCY_KEY OR
       NOT "${BLACKFRAME_OPENIMAGEIO_DEPENDENCY_KEY}" STREQUAL
           "${openimageio_local_dependency_key}")
        get_cmake_property(openimageio_cache_variables CACHE_VARIABLES)
        foreach(openimageio_cache_variable IN LISTS openimageio_cache_variables)
            get_property(
                openimageio_cache_value
                CACHE "${openimageio_cache_variable}"
                PROPERTY VALUE
            )
            string(
                FIND
                "${openimageio_cache_value}"
                "${FETCHCONTENT_BASE_DIR}/openimageio-build/deps"
                openimageio_private_cache_position
            )
            if(NOT openimageio_private_cache_position EQUAL -1)
                unset("${openimageio_cache_variable}" CACHE)
            endif()
        endforeach()
    endif()
    set(
        BLACKFRAME_OPENIMAGEIO_DEPENDENCY_KEY
        "${openimageio_local_dependency_key}"
        CACHE INTERNAL
        "OpenImageIO private dependency and toolchain identity"
        FORCE
    )
    file(MAKE_DIRECTORY "${openimageio_local_dependency_root}")
    set(
        OpenImageIO_DEPENDENCY_TOOLCHAIN_FILE
        "${openimageio_local_dependency_root}/blackframe-toolchain.cmake"
        CACHE FILEPATH
        ""
        FORCE
    )
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BlackframeOpenImageIoToolchain.cmake.in"
        "${OpenImageIO_DEPENDENCY_TOOLCHAIN_FILE}"
        @ONLY
    )
    set(
        OpenImageIO_LOCAL_DEPS_ROOT
        "${openimageio_local_dependency_root}"
        CACHE PATH
        ""
        FORCE
    )
    set(
        openimageio_local_dependency_directory
        "${openimageio_local_dependency_root}/dist"
    )
    set(
        openimageio_local_dependency_stamp
        "${openimageio_local_dependency_root}/.blackframe-static-v2"
    )
    set(
        openimageio_local_dependency_patterns
        "include/zlib.h"
        "lib/cmake/expat-*/expat-config.cmake"
        "lib/cmake/fmt/fmt-config.cmake"
        "lib/cmake/libdeflate/libdeflate-config.cmake"
        "lib/cmake/libjpeg-turbo/libjpeg-turboConfig.cmake"
        "lib/cmake/minizip-ng/minizip-ng-config.cmake"
        "lib/cmake/OpenColorIO/OpenColorIOConfig.cmake"
        "lib/cmake/PNG/PNGConfig.cmake"
        "lib/cmake/pystring/pystringConfig.cmake"
        "lib/cmake/tiff/TiffConfig.cmake"
        "lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
        "share/cmake/tsl-robin-map/tsl-robin-mapConfig.cmake"
    )
    if(WIN32)
        list(
            APPEND
            openimageio_local_dependency_patterns
            "lib/*expat*.lib"
            "lib/*fmt*.lib"
            "lib/*deflate*.lib"
            "lib/*jpeg*.lib"
            "lib/*minizip*.lib"
            "lib/*OpenColorIO*.lib"
            "lib/*png*.lib"
            "lib/*pystring*.lib"
            "lib/*tiff*.lib"
            "lib/*yaml-cpp*.lib"
            "lib/*zlibstatic*.lib"
        )
        set(
            openimageio_forbidden_shared_dependency_patterns
            "bin/*.dll"
            "lib/*.dll"
        )
    else()
        list(
            APPEND
            openimageio_local_dependency_patterns
            "lib/lib*expat*.a"
            "lib/lib*fmt*.a"
            "lib/lib*deflate*.a"
            "lib/lib*jpeg*.a"
            "lib/lib*minizip*.a"
            "lib/lib*OpenColorIO*.a"
            "lib/lib*png*.a"
            "lib/lib*pystring*.a"
            "lib/lib*tiff*.a"
            "lib/lib*yaml-cpp*.a"
            "lib/libz.a"
        )
        set(
            openimageio_forbidden_shared_dependency_patterns
            "lib/*.dylib"
            "lib/*.so"
            "lib/*.so.*"
        )
    endif()
    # Prefer the private prefix even if the caller has package roots or a
    # system prefix configured. The closure check below remains the final
    # guard against incomplete local installs.
    list(PREPEND CMAKE_PREFIX_PATH "${openimageio_local_dependency_directory}")
    foreach(
        openimageio_local_dependency
        IN ITEMS
            ZLIB
            JPEG
            TIFF
            PNG
            OpenColorIO
            fmt
            libjpeg-turbo
            libdeflate
            minizip-ng
            pystring
            yaml-cpp
            expat
            Robinmap
    )
        set(
            ${openimageio_local_dependency}_ROOT
            "${openimageio_local_dependency_directory}"
        )
    endforeach()
    set(openimageio_local_dependency_closure_complete TRUE)
    foreach(
        openimageio_local_dependency_pattern
        IN LISTS openimageio_local_dependency_patterns
    )
        file(
            GLOB
            openimageio_local_dependency_matches
            LIST_DIRECTORIES FALSE
            "${openimageio_local_dependency_directory}/${openimageio_local_dependency_pattern}"
        )
        if(NOT openimageio_local_dependency_matches)
            set(openimageio_local_dependency_closure_complete FALSE)
            break()
        endif()
    endforeach()
    if(EXISTS "${openimageio_local_dependency_stamp}" AND
       openimageio_local_dependency_closure_complete)
        set(OpenImageIO_BUILD_LOCAL_DEPS "" CACHE STRING "" FORCE)
    else()
        set(
            OpenImageIO_BUILD_LOCAL_DEPS
            "ZLIB;libjpeg-turbo;libdeflate;TIFF;PNG;pystring;expat;yaml-cpp;minizip-ng;OpenColorIO;Robinmap;fmt"
            CACHE STRING
            ""
            FORCE
        )
    endif()
    set(OpenImageIO_DEPENDENCY_BUILD_ALLOW_UNVERIFIED_TAGS OFF CACHE BOOL "" FORCE)
    set(LOCAL_BUILD_SHARED_LIBS_DEFAULT OFF)
    set(ZLIB_BUILD_SHARED_LIBS OFF)
    set(ZLIB_USE_STATIC_LIBS ON)
    set(TIFF_BUILD_SHARED_LIBS OFF)
    set(PNG_BUILD_SHARED_LIBS OFF)
    set(OpenColorIO_BUILD_SHARED_LIBS OFF)
    set(libjpeg-turbo_BUILD_SHARED_LIBS OFF)

    # Disable optional integrations so the embedded format set cannot depend
    # on whichever SDKs happen to be installed on a builder.
    foreach(
        optional_dependency
        IN ITEMS
            JXL
            libuhdr
            Freetype
            OpenCV
            TBB
            DCMTK
            FFmpeg
            GIF
            Libheif
            LibRaw
            OpenJPEG
            openjph
            OpenVDB
            Ptex
            WebP
            R3DSDK
            Nuke
    )
        set(USE_${optional_dependency} OFF)
    endforeach()

    # FetchContent's package redirects let OIIO discover the already-created
    # build-tree targets instead of probing an installed Imath/OpenEXR copy.
    unset(Imath_DIR CACHE)
    unset(OpenEXR_DIR CACHE)
    set(Imath_VERSION "${BLACKFRAME_IMATH_VERSION}")
    set(OpenEXR_VERSION "${BLACKFRAME_OPENEXR_VERSION}")
    set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

    set(openimageio_patch_script
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BlackframePatchOpenImageIo.cmake"
    )
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${openimageio_patch_script}"
    )

    # FetchContent does not necessarily rerun PATCH_COMMAND when a populated
    # source tree survives a CMake reconfigure. Patch that tree before OIIO can
    # inspect its dependency recipes, then validate it again after population.
    set(openimageio_existing_source "${FETCHCONTENT_BASE_DIR}/openimageio-src")
    if(DEFINED FETCHCONTENT_SOURCE_DIR_OPENIMAGEIO AND
       NOT FETCHCONTENT_SOURCE_DIR_OPENIMAGEIO STREQUAL "")
        set(openimageio_existing_source "${FETCHCONTENT_SOURCE_DIR_OPENIMAGEIO}")
    endif()
    if(EXISTS "${openimageio_existing_source}/CMakeLists.txt")
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}"
                "-DOPENIMAGEIO_SOURCE_DIR=${openimageio_existing_source}"
                -P "${openimageio_patch_script}"
            RESULT_VARIABLE openimageio_patch_result
            OUTPUT_VARIABLE openimageio_patch_output
            ERROR_VARIABLE openimageio_patch_error
        )
        if(NOT openimageio_patch_result EQUAL 0)
            message(
                FATAL_ERROR
                "The existing OpenImageIO source could not be patched "
                "(${openimageio_patch_result}):\n"
                "${openimageio_patch_output}${openimageio_patch_error}"
            )
        endif()
    endif()

    FetchContent_Declare(
        openimageio
        URL
            "${BLACKFRAME_OPENIMAGEIO_URL}"
        URL_HASH
            "SHA256=${BLACKFRAME_OPENIMAGEIO_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND
            "${CMAKE_COMMAND}"
            "-DOPENIMAGEIO_SOURCE_DIR=<SOURCE_DIR>"
            -P "${openimageio_patch_script}"
        SYSTEM
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(openimageio)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}"
            "-DOPENIMAGEIO_SOURCE_DIR=${openimageio_SOURCE_DIR}"
            -P "${openimageio_patch_script}"
        RESULT_VARIABLE openimageio_patch_result
        OUTPUT_VARIABLE openimageio_patch_output
        ERROR_VARIABLE openimageio_patch_error
    )
    if(NOT openimageio_patch_result EQUAL 0)
        message(
            FATAL_ERROR
            "The populated OpenImageIO source could not be validated "
            "(${openimageio_patch_result}):\n"
            "${openimageio_patch_output}${openimageio_patch_error}"
        )
    endif()

    if(NOT TARGET OpenImageIO)
        message(
            FATAL_ERROR
            "OpenImageIO ${BLACKFRAME_OPENIMAGEIO_VERSION} did not provide OpenImageIO."
        )
    endif()
    foreach(openimageio_static_target IN ITEMS OpenImageIO OpenImageIO_Util)
        if(TARGET ${openimageio_static_target})
            get_target_property(
                openimageio_library_type
                ${openimageio_static_target}
                TYPE
            )
            if(NOT openimageio_library_type STREQUAL "STATIC_LIBRARY")
                message(
                    FATAL_ERROR
                    "The ${openimageio_static_target} dependency must be a static library."
                )
            endif()
        endif()
    endforeach()

    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Static archives do not carry Windows version resources. Avoid
        # invoking llvm-rc wrappers for metadata that cannot be consumed by
        # either archive and that may deadlock in parallel CUDA builds.
        foreach(openimageio_static_target IN ITEMS OpenImageIO OpenImageIO_Util)
            if(TARGET ${openimageio_static_target})
                get_target_property(
                    openimageio_static_sources
                    ${openimageio_static_target}
                    SOURCES
                )
                list(FILTER openimageio_static_sources EXCLUDE REGEX "\\.rc$")
                set_property(
                    TARGET ${openimageio_static_target}
                    PROPERTY SOURCES "${openimageio_static_sources}"
                )
            endif()
        endforeach()
    endif()

    # Reject stale package-cache entries from system or user prefixes. Every
    # path-valued discovery result for the locally built closure must resolve
    # below the private, toolchain-keyed install prefix.
    get_cmake_property(openimageio_cache_variables CACHE_VARIABLES)
    foreach(openimageio_cache_variable IN LISTS openimageio_cache_variables)
        if(openimageio_cache_variable MATCHES
           "^(ZLIB|PNG|JPEG|TIFF|Tiff|OpenColorIO|fmt|libjpeg-turbo|libdeflate|minizip-ng|pystring|yaml-cpp|expat|EXPAT|Robinmap|ROBINMAP)_")
            get_property(
                openimageio_cache_type
                CACHE "${openimageio_cache_variable}"
                PROPERTY TYPE
            )
            if(openimageio_cache_type STREQUAL "PATH" OR
               openimageio_cache_type STREQUAL "FILEPATH")
                get_property(
                    openimageio_cache_value
                    CACHE "${openimageio_cache_variable}"
                    PROPERTY VALUE
                )
                foreach(openimageio_cache_path IN LISTS openimageio_cache_value)
                    if(NOT openimageio_cache_path STREQUAL "" AND
                       NOT openimageio_cache_path MATCHES "-NOTFOUND$")
                        cmake_path(
                            IS_PREFIX
                            openimageio_local_dependency_directory
                            "${openimageio_cache_path}"
                            NORMALIZE
                            openimageio_cache_path_is_private
                        )
                        if(NOT openimageio_cache_path_is_private)
                            message(
                                FATAL_ERROR
                                "OpenImageIO resolved ${openimageio_cache_variable} "
                                "outside its verified private dependency prefix: "
                                "${openimageio_cache_path}"
                            )
                        endif()
                    endif()
                endforeach()
            endif()
        endif()
    endforeach()

    set(openimageio_local_dependency_closure_complete TRUE)
    foreach(
        openimageio_local_dependency_pattern
        IN LISTS openimageio_local_dependency_patterns
    )
        file(
            GLOB
            openimageio_local_dependency_matches
            LIST_DIRECTORIES FALSE
            "${openimageio_local_dependency_directory}/${openimageio_local_dependency_pattern}"
        )
        if(NOT openimageio_local_dependency_matches)
            set(openimageio_local_dependency_closure_complete FALSE)
            break()
        endif()
    endforeach()
    if(NOT openimageio_local_dependency_closure_complete)
        message(
            FATAL_ERROR
            "OpenImageIO's verified private static dependency closure is incomplete."
        )
    endif()
    foreach(
        openimageio_forbidden_shared_dependency_pattern
        IN LISTS openimageio_forbidden_shared_dependency_patterns
    )
        file(
            GLOB
            openimageio_shared_dependency_matches
            LIST_DIRECTORIES FALSE
            "${openimageio_local_dependency_directory}/${openimageio_forbidden_shared_dependency_pattern}"
        )
        if(WIN32)
            # zlib's pinned recipe installs both variants even when consumers
            # explicitly resolve zlibstatic.lib. The unused DLL is not part of
            # Blackframe's link closure; all other shared artifacts remain an
            # error.
            list(
                FILTER
                openimageio_shared_dependency_matches
                EXCLUDE
                REGEX "[/\\\\]zlib\\.dll$"
            )
        endif()
        if(openimageio_shared_dependency_matches)
            message(
                FATAL_ERROR
                "OpenImageIO's private dependency prefix contains shared libraries: "
                "${openimageio_shared_dependency_matches}"
            )
        endif()
    endforeach()
    if(NOT EXISTS "${openimageio_local_dependency_stamp}")
        file(
            WRITE
            "${openimageio_local_dependency_stamp}"
            "OpenImageIO ${BLACKFRAME_OPENIMAGEIO_VERSION} verified static dependency closure\n"
        )
    endif()

    add_library(BlackframeOpenImageIo INTERFACE)
    add_library(Blackframe::OpenImageIo ALIAS BlackframeOpenImageIo)
    target_link_libraries(BlackframeOpenImageIo INTERFACE OpenImageIO)
    target_include_directories(
        BlackframeOpenImageIo
        SYSTEM
        INTERFACE
            "${openimageio_SOURCE_DIR}/src/include"
            "${CMAKE_BINARY_DIR}/include"
    )
    blackframe_set_target_role(BlackframeOpenImageIo dependency)
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

    if(BLACKFRAME_ENABLE_OPENIMAGEIO)
        set(manifest_openimageio_enabled true)
    else()
        set(manifest_openimageio_enabled false)
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

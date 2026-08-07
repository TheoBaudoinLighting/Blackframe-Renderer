cmake_minimum_required(VERSION 3.30)

function(require_json_value expected_value)
    string(JSON actual_value ERROR_VARIABLE json_error GET "${manifest}" ${ARGN})
    if(json_error)
        message(FATAL_ERROR "Cannot read JSON path '${ARGN}': ${json_error}")
    endif()
    if(NOT actual_value STREQUAL expected_value)
        message(
            FATAL_ERROR
            "JSON path '${ARGN}' is '${actual_value}', expected '${expected_value}'."
        )
    endif()
endfunction()

function(require_fetch_dependency index name version revision sha256 enabled)
    if(enabled)
        set(expected_enabled ON)
    else()
        set(expected_enabled OFF)
    endif()

    require_json_value("${name}" dependencies "${index}" name)
    require_json_value("FetchContent" dependencies "${index}" kind)
    require_json_value("${expected_enabled}" dependencies "${index}" enabled)
    require_json_value("${version}" dependencies "${index}" version)
    require_json_value("${revision}" dependencies "${index}" revision)
    require_json_value("${sha256}" dependencies "${index}" sha256)

    string(JSON dependency_url GET "${manifest}" dependencies "${index}" url)
    string(FIND "${dependency_url}" "${revision}" revision_position)
    if(revision_position EQUAL -1)
        message(
            FATAL_ERROR
            "Dependency '${name}' URL does not contain its immutable revision: ${dependency_url}"
        )
    endif()

    string(LENGTH "${revision}" revision_length)
    if(NOT revision_length EQUAL 40 OR NOT revision MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Dependency '${name}' revision is not a full Git commit.")
    endif()
    string(LENGTH "${sha256}" sha256_length)
    if(NOT sha256_length EQUAL 64 OR NOT sha256 MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Dependency '${name}' SHA-256 is malformed.")
    endif()
endfunction()

if(NOT DEFINED MANIFEST_PATH OR NOT EXISTS "${MANIFEST_PATH}")
    message(FATAL_ERROR "Dependency manifest does not exist: ${MANIFEST_PATH}")
endif()

file(READ "${MANIFEST_PATH}" manifest)
file(SHA256 "${MANIFEST_PATH}" actual_manifest_sha256)
if(NOT actual_manifest_sha256 STREQUAL MANIFEST_SHA256)
    message(
        FATAL_ERROR
        "Dependency manifest SHA-256 is '${actual_manifest_sha256}', "
        "expected '${MANIFEST_SHA256}'."
    )
endif()

set(manifest_checksum_path "${MANIFEST_PATH}.sha256")
if(NOT EXISTS "${manifest_checksum_path}")
    message(FATAL_ERROR "Dependency manifest checksum does not exist: ${manifest_checksum_path}")
endif()
file(READ "${manifest_checksum_path}" manifest_checksum)
get_filename_component(manifest_name "${MANIFEST_PATH}" NAME)
set(expected_manifest_checksum "${actual_manifest_sha256}  ${manifest_name}\n")
if(NOT manifest_checksum STREQUAL expected_manifest_checksum)
    message(FATAL_ERROR "Dependency manifest checksum file is inconsistent.")
endif()

require_json_value("1" schema_version)
require_json_value("Blackframe" project name)
string(JSON dependency_count LENGTH "${manifest}" dependencies)
if(NOT dependency_count EQUAL 8)
    message(FATAL_ERROR "Dependency manifest contains ${dependency_count} entries, expected 8.")
endif()

require_fetch_dependency(
    0
    "GoogleTest"
    "${GOOGLETEST_VERSION}"
    "${GOOGLETEST_REVISION}"
    "${GOOGLETEST_SHA256}"
    ON
)
require_fetch_dependency(
    1
    "Google Benchmark"
    "${GOOGLE_BENCHMARK_VERSION}"
    "${GOOGLE_BENCHMARK_REVISION}"
    "${GOOGLE_BENCHMARK_SHA256}"
    "${GOOGLE_BENCHMARK_ENABLED}"
)
require_fetch_dependency(
    2
    "Embree"
    "${EMBREE_VERSION}"
    "${EMBREE_REVISION}"
    "${EMBREE_SHA256}"
    "${EMBREE_ENABLED}"
)
require_fetch_dependency(
    3
    "stb"
    "${STB_REVISION}"
    "${STB_REVISION}"
    "${STB_SHA256}"
    "${STB_ENABLED}"
)
require_fetch_dependency(
    4
    "Imath"
    "${IMATH_VERSION}"
    "${IMATH_REVISION}"
    "${IMATH_SHA256}"
    ON
)
require_fetch_dependency(
    5
    "OpenEXR"
    "${OPENEXR_VERSION}"
    "${OPENEXR_REVISION}"
    "${OPENEXR_SHA256}"
    ON
)

require_fetch_dependency(
    6
    "OpenImageIO"
    "${OPENIMAGEIO_VERSION}"
    "${OPENIMAGEIO_REVISION}"
    "${OPENIMAGEIO_SHA256}"
    "${OPENIMAGEIO_ENABLED}"
)

require_json_value("CUDA Toolkit" dependencies 7 name)
require_json_value("system" dependencies 7 kind)
if(CUDA_ENABLED)
    set(expected_cuda_enabled ON)
else()
    set(expected_cuda_enabled OFF)
endif()
require_json_value("${expected_cuda_enabled}" dependencies 7 enabled)
require_json_value("${CUDA_TOOLKIT_VERSION}" dependencies 7 version)
require_json_value("ON" dependencies 7 exact)

string(JSON manifest_cuda_architecture_count LENGTH "${manifest}" dependencies 7 architectures)
if(CUDA_ENABLED)
    require_json_value("${CUDA_TOOLKIT_VERSION}" dependencies 7 resolved_version)

    list(LENGTH CUDA_ARCHITECTURES expected_cuda_architecture_count)
    if(NOT manifest_cuda_architecture_count EQUAL expected_cuda_architecture_count)
        message(FATAL_ERROR "CUDA architecture count does not match the configured list.")
    endif()

    set(cuda_architecture_index 0)
    foreach(cuda_architecture IN LISTS CUDA_ARCHITECTURES)
        require_json_value(
            "${cuda_architecture}"
            dependencies
            7
            architectures
            "${cuda_architecture_index}"
        )
        math(EXPR cuda_architecture_index "${cuda_architecture_index} + 1")
    endforeach()
elseif(NOT manifest_cuda_architecture_count EQUAL 0)
    message(FATAL_ERROR "Disabled CUDA dependency records active architectures.")
endif()

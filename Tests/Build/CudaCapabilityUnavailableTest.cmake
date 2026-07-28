foreach(required_variable IN ITEMS MANIFEST_PATH TARGET_GRAPH_PATH)
    if(NOT DEFINED ${required_variable} OR NOT EXISTS "${${required_variable}}")
        message(FATAL_ERROR "${required_variable} does not name an existing file.")
    endif()
endforeach()

file(READ "${MANIFEST_PATH}" dependency_manifest)
string(JSON dependency_count LENGTH "${dependency_manifest}" dependencies)
if(dependency_count EQUAL 0)
    message(FATAL_ERROR "The dependency manifest contains no dependencies.")
endif()

set(cuda_dependency_index "")
math(EXPR last_dependency_index "${dependency_count} - 1")
foreach(dependency_index RANGE 0 "${last_dependency_index}")
    string(JSON dependency_name GET "${dependency_manifest}" dependencies ${dependency_index} name)
    if(dependency_name STREQUAL "CUDA Toolkit")
        set(cuda_dependency_index "${dependency_index}")
        break()
    endif()
endforeach()
if(cuda_dependency_index STREQUAL "")
    message(FATAL_ERROR "The dependency manifest does not declare the CUDA Toolkit capability.")
endif()

string(
    JSON
    cuda_enabled
    GET
    "${dependency_manifest}"
    dependencies
    ${cuda_dependency_index}
    enabled
)
if(cuda_enabled)
    message(FATAL_ERROR "The CPU-only dependency manifest reports CUDA as enabled.")
endif()

string(
    JSON
    cuda_resolved_version_type
    TYPE
    "${dependency_manifest}"
    dependencies
    ${cuda_dependency_index}
    resolved_version
)
if(NOT cuda_resolved_version_type STREQUAL "NULL")
    message(FATAL_ERROR "The CPU-only dependency manifest resolved a CUDA Toolkit version.")
endif()

string(
    JSON
    cuda_architecture_count
    LENGTH
    "${dependency_manifest}"
    dependencies
    ${cuda_dependency_index}
    architectures
)
if(NOT cuda_architecture_count EQUAL 0)
    message(FATAL_ERROR "The CPU-only dependency manifest contains CUDA architectures.")
endif()

file(READ "${TARGET_GRAPH_PATH}" target_graph)
string(JSON target_count LENGTH "${target_graph}" nodes)
set(configured_targets "")
if(target_count GREATER 0)
    math(EXPR last_target_index "${target_count} - 1")
    foreach(target_index RANGE 0 "${last_target_index}")
        string(JSON target_name GET "${target_graph}" nodes ${target_index} name)
        list(APPEND configured_targets "${target_name}")
    endforeach()
endif()

foreach(
    forbidden_target
    IN ITEMS
        BlackframeCuda
        BlackframeCudaSharedHeaders
        BlackframeCudaSmokeKernel
        BlackframeCudaToolkit
        BlackframeCudaSmokeTests
)
    if(forbidden_target IN_LIST configured_targets)
        message(FATAL_ERROR "CPU-only target graph contains '${forbidden_target}'.")
    endif()
endforeach()

message(STATUS "CUDA capability is explicitly unavailable in this CPU-only build.")

if(NOT DEFINED RENDER_EXECUTABLE OR NOT EXISTS "${RENDER_EXECUTABLE}")
    message(FATAL_ERROR "RENDER_EXECUTABLE does not name the built render executable.")
endif()

foreach(required_status IN ITEMS REFERENCE_CPU_STATUS EMBREE_STATUS CUDA_STATUS)
    if(NOT DEFINED ${required_status})
        message(FATAL_ERROR "${required_status} is required.")
    endif()
endforeach()

execute_process(
    COMMAND "${RENDER_EXECUTABLE}" --capabilities
    RESULT_VARIABLE capabilities_result
    OUTPUT_VARIABLE capability_manifest
    ERROR_VARIABLE capabilities_error
)
if(NOT capabilities_result EQUAL 0)
    message(
        FATAL_ERROR
        "render --capabilities failed with exit code ${capabilities_result}: "
        "${capabilities_error}"
    )
endif()
if(capabilities_error)
    message(FATAL_ERROR "render --capabilities wrote to stderr: ${capabilities_error}")
endif()

string(JSON schema_version GET "${capability_manifest}" schema_version)
if(NOT schema_version EQUAL 1)
    message(FATAL_ERROR "Unexpected capability manifest schema version '${schema_version}'.")
endif()

string(JSON backend_count LENGTH "${capability_manifest}" backends)
if(NOT backend_count EQUAL 3)
    message(FATAL_ERROR "Expected three registered backends, found ${backend_count}.")
endif()

set(expected_backend_ids reference_cpu cpu_embree gpu_cuda)
set(expected_backend_statuses "${REFERENCE_CPU_STATUS}" "${EMBREE_STATUS}" "${CUDA_STATUS}")
set(allowed_statuses supported experimental unavailable)

math(EXPR last_backend_index "${backend_count} - 1")
foreach(backend_index RANGE 0 "${last_backend_index}")
    string(JSON backend_member_count LENGTH "${capability_manifest}" backends ${backend_index})
    if(NOT backend_member_count EQUAL 2)
        message(FATAL_ERROR "Backend ${backend_index} contains unexpected manifest keys.")
    endif()

    string(JSON first_member MEMBER "${capability_manifest}" backends ${backend_index} 0)
    string(JSON second_member MEMBER "${capability_manifest}" backends ${backend_index} 1)
    set(actual_members "${first_member}" "${second_member}")
    foreach(required_member IN ITEMS id status)
        if(NOT required_member IN_LIST actual_members)
            message(FATAL_ERROR "Backend ${backend_index} is missing '${required_member}'.")
        endif()
    endforeach()

    string(JSON backend_id GET "${capability_manifest}" backends ${backend_index} id)
    string(JSON backend_status GET "${capability_manifest}" backends ${backend_index} status)
    list(GET expected_backend_ids ${backend_index} expected_backend_id)
    list(GET expected_backend_statuses ${backend_index} expected_backend_status)

    if(NOT backend_id STREQUAL expected_backend_id)
        message(
            FATAL_ERROR
            "Backend ${backend_index} is '${backend_id}', expected '${expected_backend_id}'."
        )
    endif()
    if(NOT backend_status IN_LIST allowed_statuses)
        message(FATAL_ERROR "Backend '${backend_id}' has unknown status '${backend_status}'.")
    endif()
    if(NOT backend_status STREQUAL expected_backend_status)
        message(
            FATAL_ERROR
            "Backend '${backend_id}' is '${backend_status}', expected "
            "'${expected_backend_status}'."
        )
    endif()
endforeach()

message(STATUS "Capability manifest is valid: ${capability_manifest}")

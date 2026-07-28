cmake_minimum_required(VERSION 3.30)

foreach(required_variable IN ITEMS PROBE_SOURCE_DIR PROBE_BINARY_DIR TARGET_GRAPH_MODULE)
    if(NOT DEFINED "${required_variable}")
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${PROBE_SOURCE_DIR}"
        -B "${PROBE_BINARY_DIR}"
        -G Ninja
        "-DTARGET_GRAPH_MODULE=${TARGET_GRAPH_MODULE}"
    RESULT_VARIABLE probe_result
    OUTPUT_VARIABLE probe_output
    ERROR_VARIABLE probe_error
)

if(probe_result EQUAL 0)
    message(FATAL_ERROR "A circular Blackframe target graph was accepted.")
endif()

string(CONCAT probe_diagnostic "${probe_output}" "${probe_error}")
if(NOT probe_diagnostic MATCHES "target graph contains a dependency cycle")
    message(
        FATAL_ERROR
        "The circular graph failed without the expected diagnostic:\n${probe_diagnostic}"
    )
endif()

if(BLACKFRAME_RUN_INCOMPATIBLE_COMPILER_PROBE)
    set(CMAKE_CXX_COMPILER_ID "IncompatibleCompiler")
    set(CMAKE_CXX_COMPILER_VERSION "1.0")
    set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_23")

    include("${BLACKFRAME_HOST_LANGUAGE_MODULE}")
    blackframe_require_host_cxx26()
    return()
endif()

if(NOT DEFINED BLACKFRAME_HOST_LANGUAGE_MODULE)
    message(FATAL_ERROR "The host-language module path is required.")
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -DBLACKFRAME_RUN_INCOMPATIBLE_COMPILER_PROBE=ON
        "-DBLACKFRAME_HOST_LANGUAGE_MODULE=${BLACKFRAME_HOST_LANGUAGE_MODULE}"
        -P
        "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE incompatible_compiler_result
    OUTPUT_VARIABLE incompatible_compiler_output
    ERROR_VARIABLE incompatible_compiler_error
)

if(incompatible_compiler_result EQUAL 0)
    message(FATAL_ERROR "A compiler without cxx_std_26 capability was accepted.")
endif()

string(
    CONCAT
    incompatible_compiler_diagnostic
    "${incompatible_compiler_output}"
    "${incompatible_compiler_error}"
)

if(NOT incompatible_compiler_diagnostic MATCHES "does not advertise" OR
   NOT incompatible_compiler_diagnostic MATCHES "cxx_std_26 compile feature")
    message(
        FATAL_ERROR
        "The incompatible compiler failed without the required C++26 diagnostic:\n"
        "${incompatible_compiler_diagnostic}"
    )
endif()

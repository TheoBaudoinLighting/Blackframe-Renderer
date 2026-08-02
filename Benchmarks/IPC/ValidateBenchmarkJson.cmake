foreach(required_variable BENCHMARK_EXECUTABLE BINARY_DIRECTORY OUTPUT_PATH BENCHMARK_FILTER)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

if(NOT EXISTS "${BENCHMARK_EXECUTABLE}")
    message(FATAL_ERROR "Benchmark executable does not exist: ${BENCHMARK_EXECUTABLE}")
endif()

set(normalized_binary_directory "${BINARY_DIRECTORY}")
set(normalized_output_path "${OUTPUT_PATH}")
cmake_path(ABSOLUTE_PATH normalized_binary_directory NORMALIZE)
cmake_path(ABSOLUTE_PATH normalized_output_path NORMALIZE)
cmake_path(
    IS_PREFIX
    normalized_binary_directory
    "${normalized_output_path}"
    NORMALIZE
    output_is_below_binary_directory
)
if(
    NOT output_is_below_binary_directory
    OR normalized_output_path STREQUAL normalized_binary_directory
)
    message(FATAL_ERROR "Benchmark JSON output must stay below the build directory.")
endif()

cmake_path(GET normalized_output_path PARENT_PATH output_directory)
file(MAKE_DIRECTORY "${output_directory}")
file(REMOVE "${normalized_output_path}")

execute_process(
    COMMAND
        "${BENCHMARK_EXECUTABLE}"
        --benchmark_min_time=0.001s
        "--benchmark_filter=${BENCHMARK_FILTER}"
        "--benchmark_out=${normalized_output_path}"
        --benchmark_out_format=json
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
)
if(NOT benchmark_result EQUAL 0)
    message(
        FATAL_ERROR
        "Benchmark execution failed with exit code ${benchmark_result}.\n"
        "stdout:\n${benchmark_stdout}\n"
        "stderr:\n${benchmark_stderr}"
    )
endif()

if(NOT EXISTS "${normalized_output_path}")
    message(FATAL_ERROR "Benchmark did not create ${normalized_output_path}.")
endif()

file(READ "${normalized_output_path}" benchmark_json)
string(JSON context_type TYPE "${benchmark_json}" context)
string(JSON benchmarks_type TYPE "${benchmark_json}" benchmarks)
if(NOT context_type STREQUAL "OBJECT")
    message(FATAL_ERROR "Benchmark JSON field 'context' must be an object.")
endif()
if(NOT benchmarks_type STREQUAL "ARRAY")
    message(FATAL_ERROR "Benchmark JSON field 'benchmarks' must be an array.")
endif()

string(JSON benchmark_count LENGTH "${benchmark_json}" benchmarks)
if(benchmark_count LESS 1)
    message(FATAL_ERROR "Benchmark JSON contains no measurements.")
endif()

set(
    required_benchmarks
    encode_ping
    decode_ping
    build_light_tree/16
    sample_light_tree/16
    query_light_tree_probability/16
    sample_power_sampler/16
    compact_stable_input/256
    compact_deterministic_path_slot/256
)
math(EXPR last_benchmark_index "${benchmark_count} - 1")
foreach(benchmark_index RANGE 0 ${last_benchmark_index})
    string(JSON benchmark_type TYPE "${benchmark_json}" benchmarks ${benchmark_index})
    if(NOT benchmark_type STREQUAL "OBJECT")
        message(FATAL_ERROR "Benchmark JSON entry ${benchmark_index} must be an object.")
    endif()

    string(JSON benchmark_name GET "${benchmark_json}" benchmarks ${benchmark_index} name)
    list(REMOVE_ITEM required_benchmarks "${benchmark_name}")

    foreach(numeric_field iterations real_time cpu_time)
        string(
            JSON
            field_type
            TYPE
            "${benchmark_json}"
            benchmarks
            ${benchmark_index}
            ${numeric_field}
        )
        if(NOT field_type STREQUAL "NUMBER")
            message(
                FATAL_ERROR
                "Benchmark '${benchmark_name}' field '${numeric_field}' must be numeric."
            )
        endif()
    endforeach()

    string(JSON time_unit_type TYPE "${benchmark_json}" benchmarks ${benchmark_index} time_unit)
    if(NOT time_unit_type STREQUAL "STRING")
        message(FATAL_ERROR "Benchmark '${benchmark_name}' field 'time_unit' must be a string.")
    endif()
endforeach()

if(required_benchmarks)
    list(JOIN required_benchmarks ", " missing_benchmarks)
    message(FATAL_ERROR "Benchmark JSON is missing: ${missing_benchmarks}.")
endif()

message(
    STATUS
    "Validated ${benchmark_count} benchmark measurements in ${normalized_output_path}"
)

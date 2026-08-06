foreach(required_variable BENCHMARK_EXECUTABLE BINARY_DIRECTORY OUTPUT_PATH)
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
    message(FATAL_ERROR "CUDA compaction benchmark JSON must stay below the build directory.")
endif()

cmake_path(GET normalized_output_path PARENT_PATH output_directory)
file(MAKE_DIRECTORY "${output_directory}")
file(REMOVE "${normalized_output_path}")

set(
    benchmark_filter
    "^cuda_wavefront_queue_compaction/(4096|65536)/(25|50|75)/real_time$"
)
execute_process(
    COMMAND
        "${BENCHMARK_EXECUTABLE}"
        --benchmark_min_time=1x
        "--benchmark_filter=${benchmark_filter}"
        "--benchmark_out=${normalized_output_path}"
        --benchmark_out_format=json
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
)
if(NOT benchmark_result EQUAL 0)
    message(
        FATAL_ERROR
        "CUDA compaction benchmark failed with exit code ${benchmark_result}.\n"
        "stdout:\n${benchmark_stdout}\n"
        "stderr:\n${benchmark_stderr}"
    )
endif()

if(NOT EXISTS "${normalized_output_path}")
    message(FATAL_ERROR "Benchmark did not create ${normalized_output_path}.")
endif()
file(SIZE "${normalized_output_path}" output_size)
if(output_size EQUAL 0)
    message(FATAL_ERROR "Benchmark created an empty CUDA compaction JSON report.")
endif()

file(READ "${normalized_output_path}" benchmark_json)
string(JSON context_type ERROR_VARIABLE context_error TYPE "${benchmark_json}" context)
string(JSON benchmarks_type ERROR_VARIABLE benchmarks_error TYPE "${benchmark_json}" benchmarks)
if(NOT context_error STREQUAL "NOTFOUND" OR NOT context_type STREQUAL "OBJECT")
    message(FATAL_ERROR "CUDA compaction JSON field 'context' must be an object.")
endif()
if(NOT benchmarks_error STREQUAL "NOTFOUND" OR NOT benchmarks_type STREQUAL "ARRAY")
    message(FATAL_ERROR "CUDA compaction JSON field 'benchmarks' must be an array.")
endif()

set(
    expected_benchmarks
    cuda_wavefront_queue_compaction/4096/25/real_time
    cuda_wavefront_queue_compaction/4096/50/real_time
    cuda_wavefront_queue_compaction/4096/75/real_time
    cuda_wavefront_queue_compaction/65536/25/real_time
    cuda_wavefront_queue_compaction/65536/50/real_time
    cuda_wavefront_queue_compaction/65536/75/real_time
)
string(JSON benchmark_count LENGTH "${benchmark_json}" benchmarks)
list(LENGTH expected_benchmarks expected_benchmark_count)
if(NOT benchmark_count EQUAL expected_benchmark_count)
    message(
        FATAL_ERROR
        "Expected ${expected_benchmark_count} CUDA compaction measurements, got "
        "${benchmark_count}."
    )
endif()

function(read_required_number json index field output_variable)
    string(
        JSON field_type
        ERROR_VARIABLE field_lookup
        TYPE
        "${json}"
        benchmarks
        ${index}
        "${field}"
    )
    if(NOT field_lookup STREQUAL "NOTFOUND" OR NOT field_type STREQUAL "NUMBER")
        message(FATAL_ERROR "CUDA compaction field '${field}' must be numeric.")
    endif()
    string(JSON field_value GET "${json}" benchmarks ${index} "${field}")
    if(field_value MATCHES "^-?1[eE]\\+9999$")
        message(FATAL_ERROR "CUDA compaction field '${field}' must be finite.")
    endif()
    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

math(EXPR last_benchmark_index "${benchmark_count} - 1")
foreach(benchmark_index RANGE 0 ${last_benchmark_index})
    string(JSON benchmark_type TYPE "${benchmark_json}" benchmarks ${benchmark_index})
    if(NOT benchmark_type STREQUAL "OBJECT")
        message(FATAL_ERROR "CUDA compaction entry ${benchmark_index} must be an object.")
    endif()

    string(JSON benchmark_name GET "${benchmark_json}" benchmarks ${benchmark_index} name)
    if(NOT benchmark_name IN_LIST expected_benchmarks)
        message(FATAL_ERROR "Unexpected CUDA compaction benchmark '${benchmark_name}'.")
    endif()
    list(REMOVE_ITEM expected_benchmarks "${benchmark_name}")

    if(
        NOT benchmark_name
            MATCHES
            "^cuda_wavefront_queue_compaction/(4096|65536)/(25|50|75)/real_time$"
    )
        message(FATAL_ERROR "Malformed CUDA compaction benchmark name '${benchmark_name}'.")
    endif()
    set(expected_input_lanes "${CMAKE_MATCH_1}")
    set(occupancy_percent "${CMAKE_MATCH_2}")
    math(EXPR expected_selected_lanes "${expected_input_lanes} * ${occupancy_percent} / 100")

    string(
        JSON error_occurred_type
        ERROR_VARIABLE error_occurred_lookup
        TYPE
        "${benchmark_json}"
        benchmarks
        ${benchmark_index}
        error_occurred
    )
    if(error_occurred_lookup STREQUAL "NOTFOUND")
        if(NOT error_occurred_type STREQUAL "BOOLEAN")
            message(FATAL_ERROR "CUDA compaction field 'error_occurred' must be boolean.")
        endif()
        string(
            JSON error_occurred
            GET
            "${benchmark_json}"
            benchmarks
            ${benchmark_index}
            error_occurred
        )
        if(error_occurred)
            message(FATAL_ERROR "CUDA compaction benchmark '${benchmark_name}' reported an error.")
        endif()
    endif()

    foreach(
        numeric_field
        IN ITEMS
            iterations
            real_time
            cpu_time
            input_lanes
            selected_lanes
            published_lanes
            rejected_lanes
            scratch_bytes
            checksum
    )
        read_required_number("${benchmark_json}" ${benchmark_index} "${numeric_field}" field_value)
        if(field_value LESS 0)
            message(
                FATAL_ERROR
                "CUDA compaction field '${numeric_field}' must be non-negative for "
                "'${benchmark_name}'."
            )
        endif()
        set("${numeric_field}" "${field_value}")
    endforeach()

    if(NOT input_lanes EQUAL expected_input_lanes)
        message(FATAL_ERROR "CUDA compaction input count disagrees with '${benchmark_name}'.")
    endif()
    if(NOT selected_lanes EQUAL expected_selected_lanes)
        message(FATAL_ERROR "CUDA compaction selected count disagrees with '${benchmark_name}'.")
    endif()
    if(NOT published_lanes EQUAL selected_lanes)
        message(FATAL_ERROR "CUDA compaction failed to publish every selected lane.")
    endif()
    if(NOT rejected_lanes EQUAL 0)
        message(FATAL_ERROR "CUDA compaction unexpectedly rejected lanes.")
    endif()
    if(scratch_bytes LESS_EQUAL 0)
        message(FATAL_ERROR "CUDA compaction must report reusable scratch storage.")
    endif()
    if(checksum LESS_EQUAL 0)
        message(FATAL_ERROR "CUDA compaction checksum must be positive.")
    endif()

    string(JSON time_unit_type TYPE "${benchmark_json}" benchmarks ${benchmark_index} time_unit)
    if(NOT time_unit_type STREQUAL "STRING")
        message(FATAL_ERROR "CUDA compaction field 'time_unit' must be a string.")
    endif()
endforeach()

if(expected_benchmarks)
    list(JOIN expected_benchmarks ", " missing_benchmarks)
    message(FATAL_ERROR "CUDA compaction JSON is missing: ${missing_benchmarks}.")
endif()

message(
    STATUS
    "Validated ${benchmark_count} CUDA wavefront compaction measurements in "
    "${normalized_output_path}"
)

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
    message(FATAL_ERROR "Wavefront queue statistics JSON must stay below the build directory.")
endif()

cmake_path(GET normalized_output_path PARENT_PATH output_directory)
file(MAKE_DIRECTORY "${output_directory}")
file(REMOVE "${normalized_output_path}")

set(benchmark_name cornell_wavefront_queue_statistics)
execute_process(
    COMMAND
        "${BENCHMARK_EXECUTABLE}"
        --benchmark_min_time=1x
        "--benchmark_filter=^${benchmark_name}$"
        "--benchmark_out=${normalized_output_path}"
        --benchmark_out_format=json
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
)
if(NOT benchmark_result EQUAL 0)
    message(
        FATAL_ERROR
        "Wavefront queue statistics benchmark failed with exit code ${benchmark_result}.\n"
        "stdout:\n${benchmark_stdout}\n"
        "stderr:\n${benchmark_stderr}"
    )
endif()

if(NOT EXISTS "${normalized_output_path}")
    message(FATAL_ERROR "Benchmark did not create ${normalized_output_path}.")
endif()
file(SIZE "${normalized_output_path}" output_size)
if(output_size EQUAL 0)
    message(FATAL_ERROR "Benchmark created an empty wavefront queue statistics report.")
endif()

file(READ "${normalized_output_path}" benchmark_json)
string(JSON context_type ERROR_VARIABLE context_error TYPE "${benchmark_json}" context)
string(JSON benchmarks_type ERROR_VARIABLE benchmarks_error TYPE "${benchmark_json}" benchmarks)
if(NOT context_error STREQUAL "NOTFOUND" OR NOT context_type STREQUAL "OBJECT")
    message(FATAL_ERROR "Wavefront queue statistics JSON field 'context' must be an object.")
endif()
if(NOT benchmarks_error STREQUAL "NOTFOUND" OR NOT benchmarks_type STREQUAL "ARRAY")
    message(FATAL_ERROR "Wavefront queue statistics JSON field 'benchmarks' must be an array.")
endif()

string(JSON benchmark_count LENGTH "${benchmark_json}" benchmarks)
if(NOT benchmark_count EQUAL 1)
    message(
        FATAL_ERROR
        "The exact wavefront queue statistics filter must produce one entry, got "
        "${benchmark_count}."
    )
endif()

string(JSON actual_benchmark_name GET "${benchmark_json}" benchmarks 0 name)
if(NOT actual_benchmark_name STREQUAL benchmark_name)
    message(FATAL_ERROR "Expected benchmark '${benchmark_name}', got '${actual_benchmark_name}'.")
endif()

string(
    JSON error_occurred_type
    ERROR_VARIABLE error_occurred_lookup
    TYPE
    "${benchmark_json}"
    benchmarks
    0
    error_occurred
)
if(error_occurred_lookup STREQUAL "NOTFOUND")
    if(NOT error_occurred_type STREQUAL "BOOLEAN")
        message(FATAL_ERROR "Benchmark field 'error_occurred' must be boolean when present.")
    endif()
    string(JSON error_occurred GET "${benchmark_json}" benchmarks 0 error_occurred)
    if(error_occurred)
        string(
            JSON benchmark_error_message
            ERROR_VARIABLE error_message_lookup
            GET
            "${benchmark_json}"
            benchmarks
            0
            error_message
        )
        if(NOT error_message_lookup STREQUAL "NOTFOUND")
            set(benchmark_error_message "No benchmark error message was reported.")
        endif()
        message(FATAL_ERROR "Queue statistics benchmark reported failure: ${benchmark_error_message}")
    endif()
endif()

function(read_required_number field output_variable)
    string(
        JSON field_type
        ERROR_VARIABLE field_lookup
        TYPE
        "${benchmark_json}"
        benchmarks
        0
        "${field}"
    )
    if(NOT field_lookup STREQUAL "NOTFOUND" OR NOT field_type STREQUAL "NUMBER")
        message(FATAL_ERROR "Queue statistics field '${field}' must be numeric.")
    endif()
    string(JSON field_value GET "${benchmark_json}" benchmarks 0 "${field}")
    if(field_value MATCHES "^-?1[eE]\\+9999$")
        message(FATAL_ERROR "Queue statistics field '${field}' must be finite.")
    endif()
    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

function(read_required_nonnegative_number field output_variable)
    read_required_number("${field}" field_value)
    if(field_value LESS 0)
        message(FATAL_ERROR "Queue statistics field '${field}' must be non-negative.")
    endif()
    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

function(normalize_nonnegative_integer field_value field output_variable)
    if(
        NOT "${field_value}"
            MATCHES
            "^([0-9]+)(\\.([0-9]+))?([eE]([+-]?)([0-9]+))?$"
    )
        message(FATAL_ERROR "Queue statistics field '${field}' must be a non-negative integer.")
    endif()

    set(integer_digits "${CMAKE_MATCH_1}")
    set(fraction_digits "${CMAKE_MATCH_3}")
    set(exponent_sign "${CMAKE_MATCH_5}")
    set(exponent_digits "${CMAKE_MATCH_6}")
    set(exponent 0)
    if(NOT exponent_digits STREQUAL "")
        string(REGEX REPLACE "^0+" "" trimmed_exponent "${exponent_digits}")
        if(NOT trimmed_exponent STREQUAL "")
            string(LENGTH "${trimmed_exponent}" exponent_digit_count)
            if(exponent_digit_count GREATER 3)
                message(
                    FATAL_ERROR
                    "Queue statistics field '${field}' exceeds the validator integer domain."
                )
            endif()
            math(EXPR exponent "${trimmed_exponent}")
        endif()
        if(exponent_sign STREQUAL "-")
            math(EXPR exponent "-${exponent}")
        endif()
    endif()

    set(coefficient "${integer_digits}${fraction_digits}")
    string(LENGTH "${integer_digits}" integer_digit_count)
    string(LENGTH "${coefficient}" coefficient_digit_count)
    math(EXPR decimal_index "${integer_digit_count} + ${exponent}")

    if(decimal_index LESS_EQUAL 0)
        if(NOT coefficient MATCHES "^0+$")
            message(FATAL_ERROR "Queue statistics field '${field}' must be an integer.")
        endif()
        set(normalized_value 0)
    elseif(decimal_index LESS coefficient_digit_count)
        math(EXPR fractional_tail_length "${coefficient_digit_count} - ${decimal_index}")
        string(
            SUBSTRING
            "${coefficient}"
            ${decimal_index}
            ${fractional_tail_length}
            fractional_tail
        )
        if(NOT fractional_tail MATCHES "^0+$")
            message(FATAL_ERROR "Queue statistics field '${field}' must be an integer.")
        endif()
        string(SUBSTRING "${coefficient}" 0 ${decimal_index} normalized_value)
    elseif(decimal_index EQUAL coefficient_digit_count)
        set(normalized_value "${coefficient}")
    else()
        math(EXPR trailing_zero_count "${decimal_index} - ${coefficient_digit_count}")
        if(trailing_zero_count GREATER 18)
            message(
                FATAL_ERROR
                "Queue statistics field '${field}' exceeds the validator integer domain."
            )
        endif()
        string(REPEAT "0" ${trailing_zero_count} trailing_zeros)
        set(normalized_value "${coefficient}${trailing_zeros}")
    endif()

    string(REGEX REPLACE "^0+" "" normalized_value "${normalized_value}")
    if(normalized_value STREQUAL "")
        set(normalized_value 0)
    endif()
    string(LENGTH "${normalized_value}" normalized_digit_count)
    if(normalized_digit_count GREATER 18)
        message(
            FATAL_ERROR
            "Queue statistics field '${field}' exceeds the validator integer domain."
        )
    endif()
    set("${output_variable}" "${normalized_value}" PARENT_SCOPE)
endfunction()

function(read_required_nonnegative_integer field output_variable)
    read_required_number("${field}" field_value)
    normalize_nonnegative_integer("${field_value}" "${field}" normalized_value)
    set("${output_variable}" "${normalized_value}" PARENT_SCOPE)
endfunction()

read_required_nonnegative_integer(iterations iterations)
if(NOT iterations STREQUAL "1")
    message(FATAL_ERROR "Wavefront queue statistics must execute exactly one iteration.")
endif()
read_required_nonnegative_number(real_time real_time)
read_required_nonnegative_number(cpu_time cpu_time)
string(JSON time_unit_type TYPE "${benchmark_json}" benchmarks 0 time_unit)
if(NOT time_unit_type STREQUAL "STRING")
    message(FATAL_ERROR "Benchmark field 'time_unit' must be a string.")
endif()

read_required_nonnegative_integer(report_schema_version report_schema_version)
read_required_nonnegative_integer(configured_workers configured_workers)
read_required_nonnegative_integer(path_count path_count)
read_required_nonnegative_integer(closure_samples closure_samples)
read_required_nonnegative_integer(light_samples light_samples)
read_required_nonnegative_integer(shadow_queries shadow_queries)
read_required_nonnegative_integer(total_stage_wall_nanoseconds total_stage_wall_nanoseconds)

if(NOT report_schema_version STREQUAL "2")
    message(FATAL_ERROR "Wavefront queue statistics report schema must be version 2.")
endif()
if(NOT configured_workers STREQUAL "4")
    message(FATAL_ERROR "Wavefront queue statistics must use exactly four workers.")
endif()
if(NOT path_count STREQUAL "16384")
    message(FATAL_ERROR "Wavefront queue statistics must contain exactly 16384 paths.")
endif()

set(stage_prefixes camera ray hit miss shade shadow continuation)
set(stage_wall_values)
foreach(stage_prefix IN LISTS stage_prefixes)
    foreach(
        integer_suffix
        IN ITEMS
            capacity
            peak_size
            dispatch_count
            input_lanes
            overflow_attempts
            rejected_lanes
            stage_wall_nanoseconds
    )
        set(field_name "${stage_prefix}_${integer_suffix}")
        read_required_nonnegative_integer("${field_name}" field_value)
        set("${field_name}" "${field_value}")
    endforeach()

    foreach(occupancy_suffix IN ITEMS peak_occupancy mean_occupancy)
        set(field_name "${stage_prefix}_${occupancy_suffix}")
        read_required_nonnegative_number("${field_name}" field_value)
        if(field_value GREATER 1)
            message(FATAL_ERROR "Queue statistics field '${field_name}' must be in [0, 1].")
        endif()
        set("${field_name}" "${field_value}")
    endforeach()

    set(capacity_variable "${stage_prefix}_capacity")
    set(peak_size_variable "${stage_prefix}_peak_size")
    set(overflow_variable "${stage_prefix}_overflow_attempts")
    set(rejected_variable "${stage_prefix}_rejected_lanes")
    set(stage_wall_variable "${stage_prefix}_stage_wall_nanoseconds")
    if(NOT "${${capacity_variable}}" STREQUAL "${path_count}")
        message(FATAL_ERROR "Queue '${stage_prefix}' capacity must equal path_count.")
    endif()
    if(${peak_size_variable} GREATER ${capacity_variable})
        message(FATAL_ERROR "Queue '${stage_prefix}' peak_size exceeds its capacity.")
    endif()
    if(NOT "${${overflow_variable}}" STREQUAL "0")
        message(FATAL_ERROR "Queue '${stage_prefix}' reported an overflow attempt.")
    endif()
    if(NOT "${${rejected_variable}}" STREQUAL "0")
        message(FATAL_ERROR "Queue '${stage_prefix}' rejected lanes.")
    endif()
    list(APPEND stage_wall_values "${${stage_wall_variable}}")
endforeach()

if(
    NOT camera_input_lanes STREQUAL path_count
    OR NOT camera_peak_size STREQUAL path_count
    OR NOT camera_dispatch_count STREQUAL "1"
)
    message(
        FATAL_ERROR
        "The camera queue must dispatch and peak at the complete path batch exactly once."
    )
endif()

math(EXPR expected_ray_input_lanes "${camera_input_lanes} + ${continuation_input_lanes}")
if(NOT ray_input_lanes STREQUAL "${expected_ray_input_lanes}")
    message(FATAL_ERROR "Ray input lanes must equal camera plus continuation input lanes.")
endif()
math(EXPR classified_ray_input_lanes "${hit_input_lanes} + ${miss_input_lanes}")
if(NOT ray_input_lanes STREQUAL "${classified_ray_input_lanes}")
    message(FATAL_ERROR "Every ray input lane must be classified as a hit or miss.")
endif()
if(NOT hit_input_lanes STREQUAL shade_input_lanes)
    message(FATAL_ERROR "Every hit input lane must reach the shade queue.")
endif()
if(NOT shadow_input_lanes STREQUAL shadow_queries)
    message(FATAL_ERROR "Every shadow input lane must issue exactly one shadow query.")
endif()

list(JOIN stage_wall_values " + " stage_wall_expression)
math(EXPR summed_stage_wall_nanoseconds "${stage_wall_expression}")
if(summed_stage_wall_nanoseconds LESS_EQUAL 0)
    message(FATAL_ERROR "The sum of stage wall times must be positive.")
endif()
if(NOT total_stage_wall_nanoseconds STREQUAL "${summed_stage_wall_nanoseconds}")
    message(FATAL_ERROR "Total stage wall time must equal the sum of all seven stage times.")
endif()

message(
    STATUS
    "Validated ${benchmark_name}: ${path_count} paths, ${configured_workers} workers, "
    "${total_stage_wall_nanoseconds} ns across seven queues"
)

foreach(required_variable BENCHMARK_EXECUTABLE BINARY_DIRECTORY OUTPUT_PATH PNG_PATH)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

if(NOT EXISTS "${BENCHMARK_EXECUTABLE}")
    message(FATAL_ERROR "Benchmark executable does not exist: ${BENCHMARK_EXECUTABLE}")
endif()

set(normalized_binary_directory "${BINARY_DIRECTORY}")
set(normalized_output_path "${OUTPUT_PATH}")
set(normalized_png_path "${PNG_PATH}")
cmake_path(ABSOLUTE_PATH normalized_binary_directory NORMALIZE)
cmake_path(ABSOLUTE_PATH normalized_output_path NORMALIZE)
cmake_path(ABSOLUTE_PATH normalized_png_path NORMALIZE)

foreach(artifact_path normalized_output_path normalized_png_path)
    cmake_path(
        IS_PREFIX
        normalized_binary_directory
        "${${artifact_path}}"
        NORMALIZE
        artifact_is_below_binary_directory
    )
    if(
        NOT artifact_is_below_binary_directory
        OR "${${artifact_path}}" STREQUAL normalized_binary_directory
    )
        message(FATAL_ERROR "Benchmark artifacts must stay below the build directory.")
    endif()
endforeach()

set(
    expected_png_path
    "${normalized_binary_directory}/Benchmarks/Renderer/artifacts/cornell-power-mis-256spp.png"
)
cmake_path(NORMAL_PATH expected_png_path)
if(NOT normalized_png_path STREQUAL expected_png_path)
    message(FATAL_ERROR "The convergence preview path is not the canonical build artifact path.")
endif()

cmake_path(GET normalized_output_path PARENT_PATH output_directory)
cmake_path(GET normalized_png_path PARENT_PATH png_directory)
file(MAKE_DIRECTORY "${output_directory}" "${png_directory}")
file(REMOVE "${normalized_output_path}" "${normalized_png_path}")

set(benchmark_name cornell_power_mis_convergence)
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
        "Convergence benchmark failed with exit code ${benchmark_result}.\n"
        "stdout:\n${benchmark_stdout}\n"
        "stderr:\n${benchmark_stderr}"
    )
endif()

if(NOT EXISTS "${normalized_output_path}")
    message(FATAL_ERROR "Convergence benchmark did not create ${normalized_output_path}.")
endif()
file(SIZE "${normalized_output_path}" output_size)
if(output_size EQUAL 0)
    message(FATAL_ERROR "Convergence benchmark created an empty JSON report.")
endif()

file(READ "${normalized_output_path}" benchmark_json)
string(JSON context_type ERROR_VARIABLE context_error TYPE "${benchmark_json}" context)
string(JSON benchmarks_type ERROR_VARIABLE benchmarks_error TYPE "${benchmark_json}" benchmarks)
if(NOT context_error STREQUAL "NOTFOUND" OR NOT context_type STREQUAL "OBJECT")
    message(FATAL_ERROR "Convergence benchmark JSON field 'context' must be an object.")
endif()
if(NOT benchmarks_error STREQUAL "NOTFOUND" OR NOT benchmarks_type STREQUAL "ARRAY")
    message(FATAL_ERROR "Convergence benchmark JSON field 'benchmarks' must be an array.")
endif()

string(JSON benchmark_count LENGTH "${benchmark_json}" benchmarks)
if(NOT benchmark_count EQUAL 1)
    message(
        FATAL_ERROR
        "The exact convergence filter must produce one benchmark entry, got ${benchmark_count}."
    )
endif()

string(JSON actual_benchmark_name GET "${benchmark_json}" benchmarks 0 name)
if(NOT actual_benchmark_name STREQUAL benchmark_name)
    message(
        FATAL_ERROR
        "Expected benchmark '${benchmark_name}', got '${actual_benchmark_name}'."
    )
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
        message(FATAL_ERROR "Convergence benchmark reported failure: ${benchmark_error_message}")
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
        message(FATAL_ERROR "Convergence benchmark field '${field}' must be numeric.")
    endif()
    string(JSON field_value GET "${benchmark_json}" benchmarks 0 "${field}")
    if(field_value MATCHES "^-?1e\\+9999$")
        message(FATAL_ERROR "Convergence benchmark field '${field}' must be finite.")
    endif()
    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

function(require_nonnegative_number field output_variable)
    read_required_number("${field}" field_value)
    if(field_value LESS 0)
        message(FATAL_ERROR "Convergence benchmark field '${field}' must be non-negative.")
    endif()
    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

function(require_positive_integer field output_variable)
    read_required_number("${field}" field_value)
    if(field_value LESS_EQUAL 0)
        message(FATAL_ERROR "Convergence benchmark field '${field}' must be positive.")
    endif()

    if(NOT "${field_value}" MATCHES "^([0-9]+)(\\.([0-9]+))?([eE]([+-]?)([0-9]+))?$")
        message(FATAL_ERROR "Convergence benchmark field '${field}' must be a positive integer.")
    endif()
    set(integer_digits "${CMAKE_MATCH_1}")
    set(fraction_digits "${CMAKE_MATCH_3}")
    set(exponent_sign "${CMAKE_MATCH_5}")
    set(exponent_digits "${CMAKE_MATCH_6}")

    set(exponent 0)
    if(NOT exponent_digits STREQUAL "")
        string(LENGTH "${exponent_digits}" exponent_digit_count)
        if(exponent_digit_count GREATER 6)
            message(
                FATAL_ERROR
                "Convergence benchmark field '${field}' has an unsupported integer exponent."
            )
        endif()
        string(REGEX REPLACE "^0+" "" exponent_digits "${exponent_digits}")
        if(NOT exponent_digits STREQUAL "")
            math(EXPR exponent "${exponent_digits}")
        endif()
        if(exponent_sign STREQUAL "-")
            math(EXPR exponent "-${exponent}")
        endif()
    endif()

    string(LENGTH "${fraction_digits}" fractional_digit_count)
    math(EXPR required_trailing_zeros "${fractional_digit_count} - ${exponent}")
    if(required_trailing_zeros GREATER 0)
        set(coefficient "${integer_digits}${fraction_digits}")
        string(LENGTH "${coefficient}" coefficient_length)
        if(required_trailing_zeros GREATER coefficient_length)
            message(
                FATAL_ERROR
                "Convergence benchmark field '${field}' must be a positive integer."
            )
        endif()
        math(EXPR trailing_zero_offset "${coefficient_length} - ${required_trailing_zeros}")
        string(
            SUBSTRING
            "${coefficient}"
            "${trailing_zero_offset}"
            "${required_trailing_zeros}"
            trailing_digits
        )
        if(NOT trailing_digits MATCHES "^0+$")
            message(
                FATAL_ERROR
                "Convergence benchmark field '${field}' must be a positive integer."
            )
        endif()
    endif()

    set("${output_variable}" "${field_value}" PARENT_SCOPE)
endfunction()

require_positive_integer(iterations iterations)
require_nonnegative_number(real_time real_time)
require_nonnegative_number(cpu_time cpu_time)

string(JSON time_unit_type TYPE "${benchmark_json}" benchmarks 0 time_unit)
if(NOT time_unit_type STREQUAL "STRING")
    message(FATAL_ERROR "Convergence benchmark field 'time_unit' must be a string.")
endif()

require_nonnegative_number(time_to_mse_seconds time_to_mse_seconds)
require_nonnegative_number(time_to_mse_mad_seconds time_to_mse_mad_seconds)
require_nonnegative_number(time_to_psnr_seconds time_to_psnr_seconds)
require_nonnegative_number(time_to_psnr_mad_seconds time_to_psnr_mad_seconds)
require_positive_integer(samples_to_mse samples_to_mse)
require_positive_integer(samples_to_mse_min samples_to_mse_min)
require_positive_integer(samples_to_mse_max samples_to_mse_max)
require_positive_integer(samples_to_psnr samples_to_psnr)
require_positive_integer(samples_to_psnr_min samples_to_psnr_min)
require_positive_integer(samples_to_psnr_max samples_to_psnr_max)
require_nonnegative_number(target_mse target_mse)
require_nonnegative_number(target_psnr target_psnr)
require_nonnegative_number(observed_mse_at_threshold observed_mse_at_threshold)
require_nonnegative_number(observed_psnr_at_threshold observed_psnr_at_threshold)
require_nonnegative_number(final_mse final_mse)
require_nonnegative_number(final_mse_mad final_mse_mad)
require_nonnegative_number(final_mse_max final_mse_max)
require_nonnegative_number(final_rmse final_rmse)
require_nonnegative_number(final_rmse_mad final_rmse_mad)
read_required_number(final_bias_mean final_bias_mean)
require_nonnegative_number(final_bias_mean_mad final_bias_mean_mad)
require_nonnegative_number(final_max_abs final_max_abs)
require_nonnegative_number(final_max_abs_mad final_max_abs_mad)
require_nonnegative_number(final_psnr final_psnr)
require_nonnegative_number(final_psnr_mad final_psnr_mad)
require_nonnegative_number(final_psnr_min final_psnr_min)
require_positive_integer(reference_spp reference_spp)
require_positive_integer(image_spp image_spp)
require_positive_integer(seed_count seed_count)
require_positive_integer(scene_instance_count scene_instance_count)
require_positive_integer(sphere_instance_count sphere_instance_count)
require_nonnegative_number(image_mean_luminance image_mean_luminance)
require_nonnegative_number(left_wall_red_ratio left_wall_red_ratio)
require_nonnegative_number(right_wall_green_ratio right_wall_green_ratio)
require_nonnegative_number(neutral_enclosure_channel_ratio neutral_enclosure_channel_ratio)
require_nonnegative_number(left_sphere_luminance_ratio left_sphere_luminance_ratio)
require_nonnegative_number(right_sphere_luminance_ratio right_sphere_luminance_ratio)

if(NOT seed_count EQUAL 8)
    message(FATAL_ERROR "The convergence report must aggregate exactly eight image seeds.")
endif()
if(NOT scene_instance_count EQUAL 8 OR NOT sphere_instance_count EQUAL 2)
    message(FATAL_ERROR "The convergence report does not describe the canonical Cornell scene.")
endif()

if(observed_mse_at_threshold GREATER target_mse)
    message(FATAL_ERROR "The reported MSE threshold crossing does not reach target_mse.")
endif()
if(observed_psnr_at_threshold LESS target_psnr)
    message(FATAL_ERROR "The reported PSNR threshold crossing does not reach target_psnr.")
endif()
if(final_mse GREATER target_mse)
    message(FATAL_ERROR "The final Cornell image does not reach target_mse.")
endif()
if(final_mse_max GREATER target_mse)
    message(FATAL_ERROR "At least one final Cornell seed does not reach target_mse.")
endif()
if(final_psnr LESS target_psnr)
    message(FATAL_ERROR "The final Cornell image does not reach target_psnr.")
endif()
if(final_psnr_min LESS target_psnr)
    message(FATAL_ERROR "At least one final Cornell seed does not reach target_psnr.")
endif()
if(samples_to_mse GREATER image_spp)
    message(FATAL_ERROR "The reported MSE sample hit exceeds image_spp.")
endif()
if(samples_to_psnr GREATER image_spp)
    message(FATAL_ERROR "The reported PSNR sample hit exceeds image_spp.")
endif()
if(
    samples_to_mse_min GREATER samples_to_mse
    OR samples_to_mse GREATER samples_to_mse_max
    OR samples_to_mse_max GREATER image_spp
)
    message(FATAL_ERROR "The MSE samples-to-quality distribution is inconsistent.")
endif()
if(
    samples_to_psnr_min GREATER samples_to_psnr
    OR samples_to_psnr GREATER samples_to_psnr_max
    OR samples_to_psnr_max GREATER image_spp
)
    message(FATAL_ERROR "The PSNR samples-to-quality distribution is inconsistent.")
endif()
if(image_mean_luminance LESS_EQUAL 0.01)
    message(FATAL_ERROR "The Cornell preview is unexpectedly black.")
endif()
if(left_wall_red_ratio LESS 2.0)
    message(FATAL_ERROR "The Cornell preview is missing its red left wall.")
endif()
if(right_wall_green_ratio LESS 1.5)
    message(FATAL_ERROR "The Cornell preview is missing its green right wall.")
endif()
if(
    neutral_enclosure_channel_ratio LESS 1.0
    OR neutral_enclosure_channel_ratio GREATER 1.5
)
    message(FATAL_ERROR "The Cornell preview enclosure is not neutral.")
endif()
if(left_sphere_luminance_ratio GREATER_EQUAL 0.65)
    message(FATAL_ERROR "The Cornell preview is missing the left sphere silhouette.")
endif()
if(right_sphere_luminance_ratio GREATER_EQUAL 0.65)
    message(FATAL_ERROR "The Cornell preview is missing the right sphere silhouette.")
endif()

if(NOT EXISTS "${normalized_png_path}")
    message(FATAL_ERROR "Convergence benchmark did not create ${normalized_png_path}.")
endif()
file(SIZE "${normalized_png_path}" png_size)
if(png_size EQUAL 0)
    message(FATAL_ERROR "Convergence benchmark created an empty PNG preview.")
endif()

message(
    STATUS
    "Validated ${benchmark_name} over ${seed_count} seeds: median MSE "
    "${observed_mse_at_threshold} at ${samples_to_mse} spp, median PSNR "
    "${observed_psnr_at_threshold} dB at ${samples_to_psnr} spp"
)

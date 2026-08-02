foreach(
    required_variable
    IN ITEMS
        BINARY_DIRECTORY
        SOURCE_DIRECTORY
        BUILD_CONFIGURATION
        EMBREE_ENABLED
        CUDA_ENABLED
        OUTPUT_PATH
        WAVEFRONT_TEST_EXECUTABLE
        NEE_TEST_EXECUTABLE
        VEACH_TEST_EXECUTABLE
        CORNELL_TEST_EXECUTABLE
)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

set(normalized_binary_directory "${BINARY_DIRECTORY}")
set(normalized_source_directory "${SOURCE_DIRECTORY}")
set(normalized_output_path "${OUTPUT_PATH}")
cmake_path(ABSOLUTE_PATH normalized_binary_directory NORMALIZE)
cmake_path(ABSOLUTE_PATH normalized_source_directory NORMALIZE)
cmake_path(ABSOLUTE_PATH normalized_output_path NORMALIZE)
if(NOT IS_DIRECTORY "${normalized_binary_directory}")
    message(FATAL_ERROR "Build directory does not exist: ${normalized_binary_directory}")
endif()
if(NOT IS_DIRECTORY "${normalized_source_directory}")
    message(FATAL_ERROR "Source directory does not exist: ${normalized_source_directory}")
endif()
if(NOT EMBREE_ENABLED MATCHES "^(ON|TRUE|1)$")
    message(FATAL_ERROR "CPU wavefront parity requires the explicit Embree option.")
endif()
if(NOT CUDA_ENABLED MATCHES "^(OFF|FALSE|0)$")
    message(FATAL_ERROR "The CPU parity report requires the explicit CUDA-disabled preset.")
endif()
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
    message(FATAL_ERROR "CPU wavefront parity JSON must stay below the build directory.")
endif()
cmake_path(GET normalized_output_path EXTENSION output_extension)
if(NOT output_extension STREQUAL ".json")
    message(FATAL_ERROR "CPU wavefront parity output must use the .json extension.")
endif()

foreach(
    executable_variable
    IN ITEMS
        WAVEFRONT_TEST_EXECUTABLE
        NEE_TEST_EXECUTABLE
        VEACH_TEST_EXECUTABLE
        CORNELL_TEST_EXECUTABLE
)
    if(NOT EXISTS "${${executable_variable}}")
        message(
            FATAL_ERROR
            "${executable_variable} does not exist: ${${executable_variable}}"
        )
    endif()
endforeach()

set(expected_scenes
    AreaEmitterBalance
    AreaEmitterPower
    EnvironmentMiss
    RussianRoulette
    SubnormalReflectance
    OccludedPointLight
    PointLightNee
    VeachMIS
    CornellDiffuse
)

set(wavefront_parity_tests
    "BalanceAndPower/WavefrontMisTransportTest.MatchesAnalyticScalarAndIsIdenticalWithOneOrManyThreads/Balance"
    "BalanceAndPower/WavefrontMisTransportTest.MatchesAnalyticScalarAndIsIdenticalWithOneOrManyThreads/Power"
    "WavefrontMisTransportQueueTest.PreservesPrimaryMissEnvironmentParityWithOneOrManyThreads"
    "WavefrontMisTransportRussianRouletteTest.MatchesAnalyticScalarAndCompactsFirstBounceTerminationsDeterministically"
    "WavefrontMisTransportNumericTest.PreservesARepresentableHighBetaTimesSubnormalLambertReflectance"
    "WavefrontMisTransportNumericTest.SkipsUnrepresentableDirectRadiometryWhenTheLightIsOccluded"
)
set(
    nee_parity_tests
    "NextEventEstimationParityTest.MatchesScalarReferenceThroughCpuWavefront"
)
set(veach_parity_tests "VeachMisParityTest.MatchesScalarReferenceThroughCpuWavefront")
set(
    cornell_parity_tests
    "CornellWavefrontParityTest.MatchesScalarReferenceThroughCpuWavefront"
)

function(read_required_json_value json expected_type context output_variable)
    set(json_path ${ARGN})
    string(
        JSON actual_type
        ERROR_VARIABLE lookup_error
        TYPE
        "${json}"
        ${json_path}
    )
    if(NOT lookup_error STREQUAL "NOTFOUND" OR NOT actual_type STREQUAL expected_type)
        message(FATAL_ERROR "${context} must be a JSON ${expected_type}.")
    endif()
    string(
        JSON value
        ERROR_VARIABLE value_error
        GET
        "${json}"
        ${json_path}
    )
    if(NOT value_error STREQUAL "NOTFOUND")
        message(FATAL_ERROR "Unable to read ${context}: ${value_error}")
    endif()
    set("${output_variable}" "${value}" PARENT_SCOPE)
endfunction()

function(require_nonnegative_integer value context)
    if(NOT "${value}" MATCHES "^(0|[1-9][0-9]*)$")
        message(FATAL_ERROR "${context} must be a non-negative integer, got '${value}'.")
    endif()
endfunction()

function(require_positive_integer value context)
    require_nonnegative_integer("${value}" "${context}")
    if("${value}" STREQUAL "0")
        message(FATAL_ERROR "${context} must be greater than zero.")
    endif()
endfunction()

function(require_nonnegative_number value context)
    string(LENGTH "${value}" value_length)
    if(
        value_length GREATER 64
        OR NOT "${value}"
               MATCHES
               "^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$"
    )
        message(FATAL_ERROR "${context} must be a finite non-negative number, got '${value}'.")
    endif()
endfunction()

function(require_finite_number value context)
    string(LENGTH "${value}" value_length)
    if(
        value_length GREATER 64
        OR NOT "${value}"
               MATCHES
               "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$"
    )
        message(FATAL_ERROR "${context} must be a finite number, got '${value}'.")
    endif()
endfunction()

function(expected_scene_for_test test_name output_variable)
    if(
        test_name STREQUAL
        "BalanceAndPower/WavefrontMisTransportTest.MatchesAnalyticScalarAndIsIdenticalWithOneOrManyThreads/Balance"
    )
        set(scene AreaEmitterBalance)
    elseif(
        test_name STREQUAL
        "BalanceAndPower/WavefrontMisTransportTest.MatchesAnalyticScalarAndIsIdenticalWithOneOrManyThreads/Power"
    )
        set(scene AreaEmitterPower)
    elseif(
        test_name STREQUAL
        "WavefrontMisTransportQueueTest.PreservesPrimaryMissEnvironmentParityWithOneOrManyThreads"
    )
        set(scene EnvironmentMiss)
    elseif(
        test_name STREQUAL
        "WavefrontMisTransportRussianRouletteTest.MatchesAnalyticScalarAndCompactsFirstBounceTerminationsDeterministically"
    )
        set(scene RussianRoulette)
    elseif(
        test_name STREQUAL
        "WavefrontMisTransportNumericTest.PreservesARepresentableHighBetaTimesSubnormalLambertReflectance"
    )
        set(scene SubnormalReflectance)
    elseif(
        test_name STREQUAL
        "WavefrontMisTransportNumericTest.SkipsUnrepresentableDirectRadiometryWhenTheLightIsOccluded"
    )
        set(scene OccludedPointLight)
    elseif(
        test_name STREQUAL
        "NextEventEstimationParityTest.MatchesScalarReferenceThroughCpuWavefront"
    )
        set(scene PointLightNee)
    elseif(test_name STREQUAL "VeachMisParityTest.MatchesScalarReferenceThroughCpuWavefront")
        set(scene VeachMIS)
    elseif(
        test_name STREQUAL
        "CornellWavefrontParityTest.MatchesScalarReferenceThroughCpuWavefront"
    )
        set(scene CornellDiffuse)
    else()
        message(FATAL_ERROR "Unexpected CPU wavefront parity test '${test_name}'.")
    endif()
    set("${output_variable}" "${scene}" PARENT_SCOPE)
endfunction()

function(validate_parity_case gtest_json suite_index case_index test_name)
    set(case_path testsuites ${suite_index} testsuite ${case_index})
    set(context "CPU wavefront parity test '${test_name}'")
    expected_scene_for_test("${test_name}" expected_scene)

    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'parity_schema_version'"
        parity_schema_version
        ${case_path}
        parity_schema_version
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'wavefront_report_schema_version'"
        wavefront_report_schema_version
        ${case_path}
        wavefront_report_schema_version
    )
    read_required_json_value(
        "${gtest_json}" STRING "${context} property 'scene'" scene ${case_path} scene
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'scalar_backend'"
        scalar_backend
        ${case_path}
        scalar_backend
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'scalar_acceleration'"
        scalar_acceleration
        ${case_path}
        scalar_acceleration
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'scalar_transport_precision'"
        scalar_transport_precision
        ${case_path}
        scalar_transport_precision
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'scalar_accumulation_precision'"
        scalar_accumulation_precision
        ${case_path}
        scalar_accumulation_precision
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'wavefront_backend'"
        wavefront_backend
        ${case_path}
        wavefront_backend
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'wavefront_acceleration'"
        wavefront_acceleration
        ${case_path}
        wavefront_acceleration
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'wavefront_transport_precision'"
        wavefront_transport_precision
        ${case_path}
        wavefront_transport_precision
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'wavefront_accumulation_precision'"
        wavefront_accumulation_precision
        ${case_path}
        wavefront_accumulation_precision
    )
    read_required_json_value(
        "${gtest_json}" STRING "${context} property 'sampler'" sampler ${case_path} sampler
    )
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'mis_heuristic'"
        mis_heuristic
        ${case_path}
        mis_heuristic
    )

    foreach(integer_property IN ITEMS width height samples_per_pixel workers path_count)
        read_required_json_value(
            "${gtest_json}"
            STRING
            "${context} property '${integer_property}'"
            integer_value
            ${case_path}
            ${integer_property}
        )
        require_positive_integer("${integer_value}" "${context} property '${integer_property}'")
        set("${integer_property}" "${integer_value}")
    endforeach()
    foreach(
        counter_property
        IN ITEMS
            closure_samples
            light_samples
            shadow_queries
            queue_overflow_attempts
            queue_rejected_lanes
    )
        read_required_json_value(
            "${gtest_json}"
            STRING
            "${context} property '${counter_property}'"
            counter_value
            ${case_path}
            ${counter_property}
        )
        require_nonnegative_integer(
            "${counter_value}" "${context} property '${counter_property}'"
        )
        set("${counter_property}" "${counter_value}")
    endforeach()
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'bias_mean'"
        bias_mean
        ${case_path}
        bias_mean
    )
    require_finite_number("${bias_mean}" "${context} property 'bias_mean'")
    read_required_json_value(
        "${gtest_json}" STRING "${context} property 'seed'" seed ${case_path} seed
    )
    require_nonnegative_integer("${seed}" "${context} property 'seed'")

    foreach(
        metric_property
        IN ITEMS
            maximum_linear_mse
            maximum_linear_rmse
            maximum_linear_absolute_error
            maximum_path_radiance_absolute_error
            minimum_display_psnr
            mse_linear
            rmse_linear
            max_abs
            path_radiance_max_abs
    )
        read_required_json_value(
            "${gtest_json}"
            STRING
            "${context} property '${metric_property}'"
            metric_value
            ${case_path}
            ${metric_property}
        )
        require_nonnegative_number("${metric_value}" "${context} property '${metric_property}'")
        set("${metric_property}" "${metric_value}")
    endforeach()
    read_required_json_value(
        "${gtest_json}"
        STRING
        "${context} property 'psnr_display'"
        psnr_display
        ${case_path}
        psnr_display
    )
    if(NOT psnr_display STREQUAL "inf")
        require_nonnegative_number("${psnr_display}" "${context} property 'psnr_display'")
    endif()

    if(NOT parity_schema_version STREQUAL "1")
        message(
            FATAL_ERROR
            "${context} uses unsupported parity schema '${parity_schema_version}'."
        )
    endif()
    if(NOT wavefront_report_schema_version STREQUAL "2")
        message(
            FATAL_ERROR
            "${context} uses unsupported wavefront report schema "
            "'${wavefront_report_schema_version}'."
        )
    endif()
    if(NOT scene STREQUAL expected_scene)
        message(FATAL_ERROR "${context} reported scene '${scene}', expected '${expected_scene}'.")
    endif()
    if(
        NOT scalar_backend STREQUAL "scalar_ref"
        OR NOT scalar_acceleration STREQUAL "analytic_reference"
        OR NOT wavefront_backend STREQUAL "cpu_wavefront"
        OR NOT wavefront_acceleration STREQUAL "embree"
    )
        message(
            FATAL_ERROR
            "${context} must explicitly compare scalar_ref/analytic_reference with "
            "cpu_wavefront/embree."
        )
    endif()
    if(
        NOT scalar_transport_precision STREQUAL "float"
        OR NOT scalar_accumulation_precision STREQUAL "double"
        OR NOT wavefront_transport_precision STREQUAL "float"
        OR NOT wavefront_accumulation_precision STREQUAL "float"
    )
        message(
            FATAL_ERROR
            "${context} must report float transport, double scalar accumulation, and float "
            "wavefront accumulation."
        )
    endif()
    if(NOT sampler STREQUAL "independent_indexed")
        message(FATAL_ERROR "${context} did not report the indexed independent sampler.")
    endif()
    if(NOT mis_heuristic STREQUAL "balance" AND NOT mis_heuristic STREQUAL "power")
        message(FATAL_ERROR "${context} reported an unsupported MIS heuristic.")
    endif()
    if(NOT workers STREQUAL "4")
        message(FATAL_ERROR "${context} must evaluate cpu_wavefront with four workers.")
    endif()
    if(
        maximum_linear_mse LESS 1.0e-10
        OR maximum_linear_mse GREATER 1.0e-10
        OR maximum_linear_rmse LESS 1.0e-5
        OR maximum_linear_rmse GREATER 1.0e-5
        OR maximum_linear_absolute_error LESS 1.0e-4
        OR maximum_linear_absolute_error GREATER 1.0e-4
        OR maximum_path_radiance_absolute_error LESS 1.0e-4
        OR maximum_path_radiance_absolute_error GREATER 1.0e-4
        OR minimum_display_psnr LESS 80.0
        OR minimum_display_psnr GREATER 80.0
    )
        message(FATAL_ERROR "${context} did not use the strict image-parity thresholds.")
    endif()
    if(mse_linear GREATER maximum_linear_mse)
        message(
            FATAL_ERROR
            "${context} MSE ${mse_linear} exceeds ${maximum_linear_mse}."
        )
    endif()
    if(rmse_linear GREATER maximum_linear_rmse)
        message(
            FATAL_ERROR
            "${context} RMSE ${rmse_linear} exceeds ${maximum_linear_rmse}."
        )
    endif()
    if(max_abs GREATER maximum_linear_absolute_error)
        message(
            FATAL_ERROR
            "${context} image max_abs ${max_abs} exceeds ${maximum_linear_absolute_error}."
        )
    endif()
    if(path_radiance_max_abs GREATER maximum_path_radiance_absolute_error)
        message(
            FATAL_ERROR
            "${context} path radiance max_abs ${path_radiance_max_abs} exceeds "
            "${maximum_path_radiance_absolute_error}."
        )
    endif()
    if(NOT psnr_display STREQUAL "inf" AND psnr_display LESS minimum_display_psnr)
        message(
            FATAL_ERROR
            "${context} PSNR ${psnr_display} is below ${minimum_display_psnr} dB."
        )
    endif()
    if(NOT queue_overflow_attempts STREQUAL "0" OR NOT queue_rejected_lanes STREQUAL "0")
        message(FATAL_ERROR "${context} reported a wavefront queue overflow or rejected lane.")
    endif()

    get_property(scene_already_seen GLOBAL PROPERTY "BLACKFRAME_PARITY_SEEN_${scene}" SET)
    if(scene_already_seen)
        message(FATAL_ERROR "CPU wavefront parity scene '${scene}' was reported more than once.")
    endif()
    set_property(GLOBAL PROPERTY "BLACKFRAME_PARITY_SEEN_${scene}" TRUE)
    set_property(GLOBAL APPEND PROPERTY BLACKFRAME_PARITY_SCENES "${scene}")
    foreach(
        stored_property
        IN ITEMS
            test_name
            parity_schema_version
            wavefront_report_schema_version
            scalar_backend
            scalar_acceleration
            scalar_transport_precision
            scalar_accumulation_precision
            wavefront_backend
            wavefront_acceleration
            wavefront_transport_precision
            wavefront_accumulation_precision
            sampler
            mis_heuristic
            width
            height
            samples_per_pixel
            seed
            workers
            path_count
            closure_samples
            light_samples
            shadow_queries
            queue_overflow_attempts
            queue_rejected_lanes
            maximum_linear_mse
            maximum_linear_rmse
            maximum_linear_absolute_error
            maximum_path_radiance_absolute_error
            minimum_display_psnr
            mse_linear
            rmse_linear
            bias_mean
            max_abs
            path_radiance_max_abs
            psnr_display
    )
        set_property(
            GLOBAL
            PROPERTY "BLACKFRAME_PARITY_${scene}_${stored_property}"
                     "${${stored_property}}"
        )
    endforeach()
endfunction()

function(validate_gtest_report report_path expected_test_names)
    set(expected_tests ${expected_test_names})
    list(LENGTH expected_tests expected_test_count)
    if(NOT EXISTS "${report_path}")
        message(FATAL_ERROR "GoogleTest did not create ${report_path}.")
    endif()
    file(SIZE "${report_path}" report_size)
    if(report_size EQUAL 0)
        message(FATAL_ERROR "GoogleTest created an empty parity report: ${report_path}")
    endif()
    file(READ "${report_path}" gtest_json)
    string(JSON root_type ERROR_VARIABLE root_error TYPE "${gtest_json}")
    if(NOT root_error STREQUAL "NOTFOUND" OR NOT root_type STREQUAL "OBJECT")
        message(FATAL_ERROR "GoogleTest parity output must be a JSON object: ${report_path}")
    endif()

    foreach(counter IN ITEMS tests failures disabled errors)
        read_required_json_value(
            "${gtest_json}" NUMBER "GoogleTest field '${counter}'" counter_value ${counter}
        )
        require_nonnegative_integer("${counter_value}" "GoogleTest field '${counter}'")
        set("${counter}" "${counter_value}")
    endforeach()
    if(NOT tests STREQUAL "${expected_test_count}")
        message(
            FATAL_ERROR
            "Exact parity filter expected ${expected_test_count} tests, GoogleTest ran ${tests}."
        )
    endif()
    if(NOT failures STREQUAL "0" OR NOT disabled STREQUAL "0" OR NOT errors STREQUAL "0")
        message(FATAL_ERROR "GoogleTest parity output contains failures, errors, or disabled tests.")
    endif()

    read_required_json_value(
        "${gtest_json}" ARRAY "GoogleTest field 'testsuites'" ignored_testsuites testsuites
    )
    string(JSON suite_count LENGTH "${gtest_json}" testsuites)
    set(found_tests)
    set(observed_test_count 0)
    if(suite_count GREATER 0)
        math(EXPR last_suite_index "${suite_count} - 1")
        foreach(suite_index RANGE 0 ${last_suite_index})
            read_required_json_value(
                "${gtest_json}"
                ARRAY
                "GoogleTest suite ${suite_index} field 'testsuite'"
                ignored_test_cases
                testsuites
                ${suite_index}
                testsuite
            )
            string(JSON case_count LENGTH "${gtest_json}" testsuites ${suite_index} testsuite)
            if(case_count GREATER 0)
                math(EXPR last_case_index "${case_count} - 1")
                foreach(case_index RANGE 0 ${last_case_index})
                    set(case_path testsuites ${suite_index} testsuite ${case_index})
                    read_required_json_value(
                        "${gtest_json}"
                        STRING
                        "GoogleTest parity case name"
                        case_name
                        ${case_path}
                        name
                    )
                    read_required_json_value(
                        "${gtest_json}"
                        STRING
                        "GoogleTest parity case classname"
                        class_name
                        ${case_path}
                        classname
                    )
                    read_required_json_value(
                        "${gtest_json}"
                        STRING
                        "GoogleTest parity case status"
                        case_status
                        ${case_path}
                        status
                    )
                    read_required_json_value(
                        "${gtest_json}"
                        STRING
                        "GoogleTest parity case result"
                        case_result
                        ${case_path}
                        result
                    )
                    set(full_test_name "${class_name}.${case_name}")
                    list(FIND expected_tests "${full_test_name}" expected_index)
                    if(expected_index EQUAL -1)
                        message(
                            FATAL_ERROR
                            "Exact parity filter unexpectedly ran '${full_test_name}'."
                        )
                    endif()
                    list(FIND found_tests "${full_test_name}" duplicate_index)
                    if(NOT duplicate_index EQUAL -1)
                        message(FATAL_ERROR "Parity test '${full_test_name}' ran more than once.")
                    endif()
                    if(NOT case_status STREQUAL "RUN" OR NOT case_result STREQUAL "COMPLETED")
                        message(FATAL_ERROR "Parity test '${full_test_name}' did not complete.")
                    endif()
                    validate_parity_case(
                        "${gtest_json}" ${suite_index} ${case_index} "${full_test_name}"
                    )
                    list(APPEND found_tests "${full_test_name}")
                    math(EXPR observed_test_count "${observed_test_count} + 1")
                endforeach()
            endif()
        endforeach()
    endif()

    if(NOT observed_test_count EQUAL expected_test_count)
        message(
            FATAL_ERROR
            "Expected ${expected_test_count} parity cases, validated ${observed_test_count}."
        )
    endif()
    foreach(expected_test IN LISTS expected_tests)
        list(FIND found_tests "${expected_test}" found_index)
        if(found_index EQUAL -1)
            message(FATAL_ERROR "Exact parity test '${expected_test}' was not executed.")
        endif()
    endforeach()
endfunction()

function(run_parity_tests executable report_name expected_test_names report_path)
    set(expected_tests ${expected_test_names})
    list(JOIN expected_tests ":" exact_filter)
    file(REMOVE "${report_path}")
    execute_process(
        COMMAND
            "${executable}"
            "--gtest_filter=${exact_filter}"
            "--gtest_output=json:${report_path}"
            --gtest_color=no
        WORKING_DIRECTORY "${normalized_binary_directory}"
        RESULT_VARIABLE test_result
        OUTPUT_VARIABLE test_stdout
        ERROR_VARIABLE test_stderr
        TIMEOUT 300
    )
    if(NOT "${test_result}" STREQUAL "0")
        message(
            FATAL_ERROR
            "${report_name} parity tests failed with result '${test_result}'.\n"
            "stdout:\n${test_stdout}\n"
            "stderr:\n${test_stderr}"
        )
    endif()
    validate_gtest_report("${report_path}" "${expected_tests}")
endfunction()

cmake_path(GET normalized_output_path PARENT_PATH output_directory)
file(MAKE_DIRECTORY "${output_directory}")
set(wavefront_gtest_json "${normalized_output_path}.wavefront.gtest.json")
set(nee_gtest_json "${normalized_output_path}.nee.gtest.json")
set(veach_gtest_json "${normalized_output_path}.veach.gtest.json")
set(cornell_gtest_json "${normalized_output_path}.cornell.gtest.json")
set(temporary_output_path "${normalized_output_path}.tmp")
file(
    REMOVE
    "${wavefront_gtest_json}"
    "${nee_gtest_json}"
    "${veach_gtest_json}"
    "${cornell_gtest_json}"
    "${temporary_output_path}"
)

run_parity_tests(
    "${WAVEFRONT_TEST_EXECUTABLE}"
    "Wavefront transport"
    "${wavefront_parity_tests}"
    "${wavefront_gtest_json}"
)
run_parity_tests(
    "${NEE_TEST_EXECUTABLE}" "Point-light NEE" "${nee_parity_tests}" "${nee_gtest_json}"
)
run_parity_tests(
    "${VEACH_TEST_EXECUTABLE}" "Veach MIS" "${veach_parity_tests}" "${veach_gtest_json}"
)
run_parity_tests(
    "${CORNELL_TEST_EXECUTABLE}"
    "Cornell diffuse"
    "${cornell_parity_tests}"
    "${cornell_gtest_json}"
)

get_property(actual_scenes GLOBAL PROPERTY BLACKFRAME_PARITY_SCENES)
list(LENGTH actual_scenes actual_scene_count)
list(LENGTH expected_scenes expected_scene_count)
if(NOT actual_scene_count EQUAL expected_scene_count)
    message(
        FATAL_ERROR
        "CPU wavefront parity inventory contains ${actual_scene_count} scenes, expected "
        "${expected_scene_count}."
    )
endif()
foreach(expected_scene IN LISTS expected_scenes)
    list(FIND actual_scenes "${expected_scene}" scene_index)
    if(scene_index EQUAL -1)
        message(FATAL_ERROR "CPU wavefront parity scene '${expected_scene}' is missing.")
    endif()
endforeach()

find_program(blackframe_git_executable NAMES git)
set(source_revision "source-archive")
set(source_revision_kind "archive_without_vcs")
if(blackframe_git_executable)
    execute_process(
        COMMAND "${blackframe_git_executable}" rev-parse --verify HEAD
        WORKING_DIRECTORY "${normalized_source_directory}"
        RESULT_VARIABLE revision_result
        OUTPUT_VARIABLE revision_output
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        TIMEOUT 10
    )
    string(TOLOWER "${revision_output}" revision_output)
    string(LENGTH "${revision_output}" revision_length)
    if(
        revision_result EQUAL 0
        AND (revision_length EQUAL 40 OR revision_length EQUAL 64)
        AND revision_output MATCHES "^[0-9a-f]+$"
    )
        set(source_revision "${revision_output}")
        set(source_revision_kind "git_commit")
    endif()
endif()

set(asset_hashes_json "{}")
foreach(
    asset
    IN ITEMS
        "cornell_scene|Tests/Backends/CPU/Embree/CornellWavefrontScene.hpp"
        "cornell_parity|Tests/Backends/CPU/Embree/CornellWavefrontParityTest.cpp"
        "point_light_nee|Tests/Backends/CPU/Embree/NextEventEstimationImageTest.cpp"
        "veach_mis|Tests/Backends/CPU/Embree/VeachMisImageTest.cpp"
        "wavefront_transport|Tests/Backends/CPU/Embree/WavefrontMisTransportTest.cpp"
)
    string(REPLACE "|" ";" asset_fields "${asset}")
    list(GET asset_fields 0 asset_name)
    list(GET asset_fields 1 asset_relative_path)
    set(asset_path "${normalized_source_directory}/${asset_relative_path}")
    if(NOT EXISTS "${asset_path}")
        message(FATAL_ERROR "Parity asset does not exist: ${asset_path}")
    endif()
    file(SHA256 "${asset_path}" asset_sha256)
    string(JSON asset_hashes_json SET "${asset_hashes_json}" "${asset_name}" "\"sha256:${asset_sha256}\"")
endforeach()

set(scenes_json "[]")
set(scene_index 0)
foreach(scene IN LISTS expected_scenes)
    foreach(
        property_name
        IN ITEMS
            test_name
            parity_schema_version
            wavefront_report_schema_version
            scalar_backend
            scalar_acceleration
            scalar_transport_precision
            scalar_accumulation_precision
            wavefront_backend
            wavefront_acceleration
            wavefront_transport_precision
            wavefront_accumulation_precision
            sampler
            mis_heuristic
            width
            height
            samples_per_pixel
            seed
            workers
            path_count
            closure_samples
            light_samples
            shadow_queries
            queue_overflow_attempts
            queue_rejected_lanes
            maximum_linear_mse
            maximum_linear_rmse
            maximum_linear_absolute_error
            maximum_path_radiance_absolute_error
            minimum_display_psnr
            mse_linear
            rmse_linear
            bias_mean
            max_abs
            path_radiance_max_abs
            psnr_display
    )
        get_property(
            "${property_name}"
            GLOBAL
            PROPERTY "BLACKFRAME_PARITY_${scene}_${property_name}"
        )
    endforeach()

    set(scene_json "{}")
    string(JSON scene_json SET "${scene_json}" scene "\"${scene}\"")
    string(JSON scene_json SET "${scene_json}" test "\"${test_name}\"")
    string(JSON scene_json SET "${scene_json}" passed true)
    string(
        JSON scene_json
        SET "${scene_json}"
        parity_schema_version
        "${parity_schema_version}"
    )
    string(
        JSON scene_json
        SET "${scene_json}"
        wavefront_report_schema_version
        "${wavefront_report_schema_version}"
    )
    string(JSON scene_json SET "${scene_json}" scalar_backend "\"${scalar_backend}\"")
    string(
        JSON scene_json
        SET "${scene_json}"
        scalar_acceleration
        "\"${scalar_acceleration}\""
    )
    string(
        JSON scene_json
        SET "${scene_json}"
        scalar_transport_precision
        "\"${scalar_transport_precision}\""
    )
    string(
        JSON scene_json
        SET "${scene_json}"
        scalar_accumulation_precision
        "\"${scalar_accumulation_precision}\""
    )
    string(JSON scene_json SET "${scene_json}" wavefront_backend "\"${wavefront_backend}\"")
    string(
        JSON scene_json
        SET "${scene_json}"
        wavefront_acceleration
        "\"${wavefront_acceleration}\""
    )
    string(
        JSON scene_json
        SET "${scene_json}"
        wavefront_transport_precision
        "\"${wavefront_transport_precision}\""
    )
    string(
        JSON scene_json
        SET "${scene_json}"
        wavefront_accumulation_precision
        "\"${wavefront_accumulation_precision}\""
    )
    string(JSON scene_json SET "${scene_json}" sampler "\"${sampler}\"")
    string(JSON scene_json SET "${scene_json}" mis_heuristic "\"${mis_heuristic}\"")
    string(JSON scene_json SET "${scene_json}" width "${width}")
    string(JSON scene_json SET "${scene_json}" height "${height}")
    string(JSON scene_json SET "${scene_json}" samples_per_pixel "${samples_per_pixel}")
    string(JSON scene_json SET "${scene_json}" seed "\"${seed}\"")
    string(JSON scene_json SET "${scene_json}" workers "${workers}")
    string(JSON scene_json SET "${scene_json}" path_count "${path_count}")

    set(counters_json "{}")
    string(JSON counters_json SET "${counters_json}" closure_samples "${closure_samples}")
    string(JSON counters_json SET "${counters_json}" light_samples "${light_samples}")
    string(JSON counters_json SET "${counters_json}" shadow_queries "${shadow_queries}")
    string(
        JSON counters_json
        SET "${counters_json}"
        queue_overflow_attempts
        "${queue_overflow_attempts}"
    )
    string(
        JSON counters_json
        SET "${counters_json}"
        queue_rejected_lanes
        "${queue_rejected_lanes}"
    )
    string(JSON scene_json SET "${scene_json}" counters "${counters_json}")

    set(metrics_json "{}")
    string(JSON metrics_json SET "${metrics_json}" mse_linear "${mse_linear}")
    string(JSON metrics_json SET "${metrics_json}" rmse_linear "${rmse_linear}")
    string(JSON metrics_json SET "${metrics_json}" bias_mean "${bias_mean}")
    string(JSON metrics_json SET "${metrics_json}" max_abs "${max_abs}")
    string(
        JSON metrics_json
        SET "${metrics_json}"
        path_radiance_max_abs
        "${path_radiance_max_abs}"
    )
    if(psnr_display STREQUAL "inf")
        string(JSON metrics_json SET "${metrics_json}" psnr_display null)
        string(JSON metrics_json SET "${metrics_json}" psnr_display_infinite true)
    else()
        string(JSON metrics_json SET "${metrics_json}" psnr_display "${psnr_display}")
        string(JSON metrics_json SET "${metrics_json}" psnr_display_infinite false)
    endif()
    string(JSON scene_json SET "${scene_json}" metrics "${metrics_json}")

    set(thresholds_json "{}")
    string(
        JSON thresholds_json
        SET "${thresholds_json}"
        maximum_linear_mse
        "${maximum_linear_mse}"
    )
    string(
        JSON thresholds_json
        SET "${thresholds_json}"
        maximum_linear_rmse
        "${maximum_linear_rmse}"
    )
    string(
        JSON thresholds_json
        SET "${thresholds_json}"
        maximum_linear_absolute_error
        "${maximum_linear_absolute_error}"
    )
    string(
        JSON thresholds_json
        SET "${thresholds_json}"
        maximum_path_radiance_absolute_error
        "${maximum_path_radiance_absolute_error}"
    )
    string(
        JSON thresholds_json
        SET "${thresholds_json}"
        minimum_display_psnr
        "${minimum_display_psnr}"
    )
    string(JSON scene_json SET "${scene_json}" thresholds "${thresholds_json}")
    string(JSON scenes_json SET "${scenes_json}" ${scene_index} "${scene_json}")
    math(EXPR scene_index "${scene_index} + 1")
endforeach()

set(report_json "{}")
string(JSON report_json SET "${report_json}" schema_version 1)
string(
    JSON report_json
    SET "${report_json}"
    suite
    "\"scalar_ref_cpu_wavefront_image_parity\""
)
string(JSON report_json SET "${report_json}" status "\"passed\"")
set(source_json "{}")
string(JSON source_json SET "${source_json}" revision "\"${source_revision}\"")
string(JSON source_json SET "${source_json}" revision_kind "\"${source_revision_kind}\"")
string(JSON report_json SET "${report_json}" source "${source_json}")

set(options_json "{}")
string(JSON options_json SET "${options_json}" build_configuration "\"${BUILD_CONFIGURATION}\"")
string(JSON options_json SET "${options_json}" embree_enabled true)
string(JSON options_json SET "${options_json}" cuda_enabled false)
string(JSON options_json SET "${options_json}" worker_count 4)
string(JSON options_json SET "${options_json}" sampler "\"independent_indexed\"")
string(JSON options_json SET "${options_json}" transport_precision "\"float\"")
string(JSON options_json SET "${options_json}" scalar_accumulation_precision "\"double\"")
string(JSON options_json SET "${options_json}" wavefront_accumulation_precision "\"float\"")
string(JSON report_json SET "${report_json}" options "${options_json}")

set(capabilities_json "[]")
set(capability_index 0)
foreach(
    capability
    IN ITEMS
        analytic_reference
        embree
        cpu_wavefront
        independent_indexed_sampler
        sampled_spectrum_4
        lambertian_closure
        punctual_lights
        mesh_area_lights
        next_event_estimation
        balance_mis
        power_mis
        russian_roulette
)
    string(JSON capabilities_json SET "${capabilities_json}" ${capability_index} "\"${capability}\"")
    math(EXPR capability_index "${capability_index} + 1")
endforeach()
string(JSON report_json SET "${report_json}" capabilities "${capabilities_json}")
string(JSON report_json SET "${report_json}" asset_hashes "${asset_hashes_json}")
string(JSON report_json SET "${report_json}" reference_backend "\"scalar_ref\"")
string(JSON report_json SET "${report_json}" evaluated_backend "\"cpu_wavefront\"")
string(JSON report_json SET "${report_json}" scene_count "${expected_scene_count}")
string(JSON report_json SET "${report_json}" scenes "${scenes_json}")

file(WRITE "${temporary_output_path}" "${report_json}\n")
file(RENAME "${temporary_output_path}" "${normalized_output_path}" RESULT rename_result)
if(NOT rename_result STREQUAL "0")
    message(FATAL_ERROR "Unable to publish parity report atomically: ${rename_result}")
endif()
file(
    REMOVE
    "${wavefront_gtest_json}"
    "${nee_gtest_json}"
    "${veach_gtest_json}"
    "${cornell_gtest_json}"
)

message(
    STATUS
    "Validated ${expected_scene_count} scalar_ref/cpu_wavefront scenes: "
    "${normalized_output_path}"
)

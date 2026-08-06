cmake_minimum_required(VERSION 3.30)

foreach(
    blackframe_required_variable
    IN ITEMS
        BLACKFRAME_NSYS_EXECUTABLE
        BLACKFRAME_NSIGHT_TARGET
        BLACKFRAME_NSIGHT_BUILD_DIRECTORY
        BLACKFRAME_NSIGHT_OUTPUT_DIRECTORY
)
    if(NOT DEFINED "${blackframe_required_variable}" OR
       "${${blackframe_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Nsight validation requires ${blackframe_required_variable}.")
    endif()
endforeach()

if(NOT EXISTS "${BLACKFRAME_NSYS_EXECUTABLE}" OR IS_DIRECTORY "${BLACKFRAME_NSYS_EXECUTABLE}")
    message(FATAL_ERROR "The configured Nsight Systems executable is unavailable.")
endif()
if(NOT EXISTS "${BLACKFRAME_NSIGHT_TARGET}" OR IS_DIRECTORY "${BLACKFRAME_NSIGHT_TARGET}")
    message(FATAL_ERROR "The CUDA Cornell smoke-test executable is unavailable.")
endif()

set(blackframe_build_directory "${BLACKFRAME_NSIGHT_BUILD_DIRECTORY}")
set(blackframe_output_directory "${BLACKFRAME_NSIGHT_OUTPUT_DIRECTORY}")
set(blackframe_target_path "${BLACKFRAME_NSIGHT_TARGET}")
cmake_path(NORMAL_PATH blackframe_build_directory)
cmake_path(NORMAL_PATH blackframe_output_directory)
cmake_path(NORMAL_PATH blackframe_target_path)
cmake_path(
    IS_PREFIX
    blackframe_build_directory
    "${blackframe_output_directory}"
    NORMALIZE
    blackframe_output_is_in_build
)
cmake_path(
    IS_PREFIX
    blackframe_build_directory
    "${blackframe_target_path}"
    NORMALIZE
    blackframe_target_is_in_build
)
if(NOT blackframe_output_is_in_build)
    message(FATAL_ERROR "Nsight validation output must remain below the build directory.")
endif()
if(NOT blackframe_target_is_in_build)
    message(FATAL_ERROR "Nsight validation may only execute a target below the build directory.")
endif()

file(MAKE_DIRECTORY "${blackframe_output_directory}")
set(blackframe_report_prefix "${blackframe_output_directory}/blackframe-cuda-wavefront")
set(blackframe_report "${blackframe_report_prefix}.nsys-rep")
set(blackframe_trace_json "${blackframe_report_prefix}.jsonl")
set(blackframe_stats_sqlite "${blackframe_report_prefix}.sqlite")
set(blackframe_nvtx_stats_json "${blackframe_report_prefix}-nvtx-summary.json")
set(blackframe_gpu_projection_json "${blackframe_report_prefix}-gpu-projection.json")
set(blackframe_kernel_stats_json "${blackframe_report_prefix}-kernel-summary.json")
set(blackframe_intermediate_report "${blackframe_report_prefix}.qdstrm")

file(
    REMOVE
        "${blackframe_report}"
        "${blackframe_trace_json}"
        "${blackframe_stats_sqlite}"
        "${blackframe_nvtx_stats_json}"
        "${blackframe_gpu_projection_json}"
        "${blackframe_kernel_stats_json}"
        "${blackframe_intermediate_report}"
)

execute_process(
    COMMAND
        "${BLACKFRAME_NSYS_EXECUTABLE}"
        profile
        --trace=cuda,nvtx
        --cuda-event-trace=true
        --sample=none
        --cpuctxsw=none
        --force-overwrite=true
        --output "${blackframe_report_prefix}"
        "${blackframe_target_path}"
        --gtest_filter=CudaCornellWavefrontSmokeTest.NsightInstrumentationExportsStageMetrics
    RESULT_VARIABLE blackframe_profile_result
    OUTPUT_VARIABLE blackframe_profile_stdout
    ERROR_VARIABLE blackframe_profile_stderr
)
if(NOT "${blackframe_profile_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Nsight Systems profiling failed with result ${blackframe_profile_result}.\n"
        "stdout:\n${blackframe_profile_stdout}\n"
        "stderr:\n${blackframe_profile_stderr}"
    )
endif()

execute_process(
    COMMAND
        "${BLACKFRAME_NSYS_EXECUTABLE}"
        export
        --type=jsonlines
        --force-overwrite=true
        --quiet=true
        --output "${blackframe_trace_json}"
        "${blackframe_report}"
    RESULT_VARIABLE blackframe_export_result
    OUTPUT_VARIABLE blackframe_export_stdout
    ERROR_VARIABLE blackframe_export_stderr
)
if(NOT "${blackframe_export_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Nsight Systems JSON export failed with result ${blackframe_export_result}.\n"
        "stdout:\n${blackframe_export_stdout}\n"
        "stderr:\n${blackframe_export_stderr}"
    )
endif()

execute_process(
    COMMAND
        "${BLACKFRAME_NSYS_EXECUTABLE}"
        stats
        --report=nvtx_sum
        --format=json
        --output=-
        --quiet
        "--sqlite=${blackframe_stats_sqlite}"
        --force-export=true
        --force-overwrite=true
        "${blackframe_report}"
    RESULT_VARIABLE blackframe_nvtx_stats_result
    OUTPUT_FILE "${blackframe_nvtx_stats_json}"
    ERROR_VARIABLE blackframe_nvtx_stats_stderr
)
if(NOT "${blackframe_nvtx_stats_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Nsight Systems NVTX statistics failed with result ${blackframe_nvtx_stats_result}.\n"
        "stderr:\n${blackframe_nvtx_stats_stderr}"
    )
endif()

execute_process(
    COMMAND
        "${BLACKFRAME_NSYS_EXECUTABLE}"
        stats
        --report=nvtx_gpu_proj_sum
        --format=json
        --output=-
        --quiet
        "--sqlite=${blackframe_stats_sqlite}"
        --force-export=true
        --force-overwrite=true
        "${blackframe_report}"
    RESULT_VARIABLE blackframe_gpu_projection_result
    OUTPUT_FILE "${blackframe_gpu_projection_json}"
    ERROR_VARIABLE blackframe_gpu_projection_stderr
)
if(NOT "${blackframe_gpu_projection_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Nsight Systems GPU projection failed with result "
        "${blackframe_gpu_projection_result}.\nstderr:\n${blackframe_gpu_projection_stderr}"
    )
endif()

execute_process(
    COMMAND
        "${BLACKFRAME_NSYS_EXECUTABLE}"
        stats
        --report=cuda_gpu_kern_sum
        --format=json
        --output=-
        --quiet
        "--sqlite=${blackframe_stats_sqlite}"
        --force-export=true
        --force-overwrite=true
        "${blackframe_report}"
    RESULT_VARIABLE blackframe_kernel_stats_result
    OUTPUT_FILE "${blackframe_kernel_stats_json}"
    ERROR_VARIABLE blackframe_kernel_stats_stderr
)
if(NOT "${blackframe_kernel_stats_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "Nsight Systems CUDA kernel statistics failed with result "
        "${blackframe_kernel_stats_result}.\nstderr:\n${blackframe_kernel_stats_stderr}"
    )
endif()

function(blackframe_require_non_empty_file blackframe_path blackframe_description)
    if(NOT EXISTS "${blackframe_path}" OR IS_DIRECTORY "${blackframe_path}")
        message(FATAL_ERROR "Nsight Systems did not produce ${blackframe_description}.")
    endif()
    file(SIZE "${blackframe_path}" blackframe_file_size)
    if(blackframe_file_size EQUAL 0)
        message(FATAL_ERROR "Nsight Systems produced an empty ${blackframe_description}.")
    endif()
endfunction()

blackframe_require_non_empty_file("${blackframe_report}" "profile report")
blackframe_require_non_empty_file("${blackframe_trace_json}" "JSON trace export")
blackframe_require_non_empty_file("${blackframe_stats_sqlite}" "SQLite statistics database")
blackframe_require_non_empty_file("${blackframe_nvtx_stats_json}" "NVTX statistics JSON")
blackframe_require_non_empty_file("${blackframe_gpu_projection_json}" "NVTX GPU projection JSON")
blackframe_require_non_empty_file("${blackframe_kernel_stats_json}" "CUDA kernel statistics JSON")

file(READ "${blackframe_gpu_projection_json}" blackframe_nvtx_validation_contents)

set(blackframe_nvtx_domain "blackframe.cuda.wavefront")
string(
    FIND
    "${blackframe_nvtx_validation_contents}"
    "\"${blackframe_nvtx_domain}:transport\""
    blackframe_nvtx_domain_position
)
if(blackframe_nvtx_domain_position EQUAL -1)
    message(FATAL_ERROR "Nsight export does not contain the expected NVTX domain.")
endif()

foreach(
    blackframe_stage_name
    IN ITEMS
        camera
        intersection
        hit
        miss
        shade
        shadow
        continuation
)
    string(
        FIND
        "${blackframe_nvtx_validation_contents}"
        "\"${blackframe_nvtx_domain}:${blackframe_stage_name}\""
        blackframe_stage_position
    )
    if(blackframe_stage_position EQUAL -1)
        message(
            FATAL_ERROR
            "Nsight export does not contain the exact NVTX stage name '${blackframe_stage_name}'."
        )
    endif()
endforeach()

message(STATUS "Nsight Systems CUDA/NVTX validation artifacts: ${blackframe_output_directory}")

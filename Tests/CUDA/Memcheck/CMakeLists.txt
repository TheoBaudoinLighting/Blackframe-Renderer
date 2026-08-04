enable_language(CUDA)
blackframe_require_cuda_cxx20()

add_executable(
    BlackframeCudaMemoryMemcheckTests
    ../CudaMemoryMemcheckKernel.cu
    ../DeviceMemoryMemcheckTest.cpp
)
blackframe_set_target_role(BlackframeCudaMemoryMemcheckTests test)
blackframe_configure_cuda_target(BlackframeCudaMemoryMemcheckTests)
set_target_properties(
    BlackframeCudaMemoryMemcheckTests
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set_property(
        TARGET BlackframeCudaMemoryMemcheckTests
        PROPERTY MSVC_DEBUG_INFORMATION_FORMAT ""
    )
endif()

target_compile_options(
    BlackframeCudaMemoryMemcheckTests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-fdevice-sanitize=memcheck>
)
target_link_options(
    BlackframeCudaMemoryMemcheckTests
    PRIVATE
        $<DEVICE_LINK:-fdevice-sanitize=memcheck>
)

target_link_libraries(
    BlackframeCudaMemoryMemcheckTests
    PRIVATE
        Blackframe::CudaMemory
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

cmake_path(GET CMAKE_CUDA_COMPILER PARENT_PATH blackframe_cuda_compiler_bin)
cmake_path(GET blackframe_cuda_compiler_bin PARENT_PATH blackframe_cuda_toolkit_root)

find_program(
    blackframe_compute_sanitizer
    NAMES compute-sanitizer compute-sanitizer.bat
    HINTS
        "${blackframe_cuda_compiler_bin}"
        "${blackframe_cuda_toolkit_root}/compute-sanitizer"
    REQUIRED
    NO_DEFAULT_PATH
    NO_CACHE
)

add_test(
    NAME BlackframeCudaMemoryMemcheck
    COMMAND
        "${blackframe_compute_sanitizer}"
        --tool memcheck
        --leak-check full
        --error-exitcode 99
        "$<TARGET_FILE:BlackframeCudaMemoryMemcheckTests>"
)
set_tests_properties(
    BlackframeCudaMemoryMemcheck
    PROPERTIES
        LABELS "nightly;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 180
)

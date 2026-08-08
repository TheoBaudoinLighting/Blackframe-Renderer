add_executable(
    BlackframeCudaSceneTests
    ConstantTextureParityTest.cpp
    ClosestHitTest.cpp
    SceneBvhTest.cpp
    SceneSoATest.cpp
)
blackframe_set_target_role(BlackframeCudaSceneTests test)

target_link_libraries(
    BlackframeCudaSceneTests
    PRIVATE
        Blackframe::CudaScene
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::HostImage)
    target_link_libraries(BlackframeCudaSceneTests PRIVATE Blackframe::HostImage)
    target_compile_definitions(
        BlackframeCudaSceneTests
        PRIVATE
            BLACKFRAME_CUDA_SURFACE_MAP_TESTS=1
            BLACKFRAME_CUDA_SURFACE_MAP_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
else()
    target_compile_definitions(
        BlackframeCudaSceneTests
        PRIVATE BLACKFRAME_CUDA_SURFACE_MAP_TESTS=0
    )
endif()

gtest_discover_tests(
    BlackframeCudaSceneTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaWavefrontQueueTests
    WavefrontQueuesTest.cpp
)
blackframe_set_target_role(BlackframeCudaWavefrontQueueTests test)

target_link_libraries(
    BlackframeCudaWavefrontQueueTests
    PRIVATE
        Blackframe::CudaQueues
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaWavefrontQueueTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaWavefrontQueueCompactionTests
    WavefrontQueueCompactionTest.cpp
)
blackframe_set_target_role(BlackframeCudaWavefrontQueueCompactionTests test)

target_link_libraries(
    BlackframeCudaWavefrontQueueCompactionTests
    PRIVATE
        Blackframe::CudaQueues
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaWavefrontQueueCompactionTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 60
)

add_executable(
    BlackframeCudaWavefrontStageAuditTests
    CudaWavefrontStageAuditTest.cpp
)
blackframe_set_target_role(BlackframeCudaWavefrontStageAuditTests test)

target_link_libraries(
    BlackframeCudaWavefrontStageAuditTests
    PRIVATE
        Blackframe::CudaMemory
        Blackframe::CudaWavefrontStageKernel
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeCudaWavefrontStageAuditTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
)

add_executable(
    BlackframeCudaCornellSmokeTests
    CudaCornellWavefrontSmokeTest.cpp
)
blackframe_set_target_role(BlackframeCudaCornellSmokeTests test)

target_link_libraries(
    BlackframeCudaCornellSmokeTests
    PRIVATE
        Blackframe::CudaWavefrontTransport
        Blackframe::CudaToolkit
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::Png)
    target_link_libraries(BlackframeCudaCornellSmokeTests PRIVATE Blackframe::Png)
    target_compile_definitions(
        BlackframeCudaCornellSmokeTests
        PRIVATE BLACKFRAME_CUDA_CORNELL_PNG=1
    )
endif()

target_compile_definitions(
    BlackframeCudaCornellSmokeTests
    PRIVATE BLACKFRAME_CUDA_TEST_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

gtest_discover_tests(
    BlackframeCudaCornellSmokeTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;regression;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 180
)

if(BLACKFRAME_ENABLE_NSIGHT_VALIDATION)
    set(
        blackframe_cuda_nsight_output_directory
        "${CMAKE_CURRENT_BINARY_DIR}/nsight/$<CONFIG>"
    )
    add_test(
        NAME BlackframeCudaNsightValidation
        COMMAND
            "${CMAKE_COMMAND}"
            "-DBLACKFRAME_NSYS_EXECUTABLE:FILEPATH=${BLACKFRAME_NSYS_EXECUTABLE}"
            "-DBLACKFRAME_NSIGHT_TARGET:FILEPATH=$<TARGET_FILE:BlackframeCudaCornellSmokeTests>"
            "-DBLACKFRAME_NSIGHT_BUILD_DIRECTORY:PATH=${CMAKE_BINARY_DIR}"
            "-DBLACKFRAME_NSIGHT_OUTPUT_DIRECTORY:PATH=${blackframe_cuda_nsight_output_directory}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/RunNsightValidation.cmake"
    )
    set_tests_properties(
        BlackframeCudaNsightValidation
        PROPERTIES
            LABELS "nightly;cuda"
            RESOURCE_LOCK blackframe_cuda_device_0
            RUN_SERIAL TRUE
            TIMEOUT 900
    )
endif()

cmake_path(GET CMAKE_CUDA_COMPILER PARENT_PATH blackframe_cuda_compiler_bin)
cmake_path(GET blackframe_cuda_compiler_bin PARENT_PATH blackframe_cuda_toolkit_root)
find_program(
    blackframe_compute_sanitizer_racecheck
    NAMES compute-sanitizer compute-sanitizer.bat
    HINTS
        "${blackframe_cuda_compiler_bin}"
        "${blackframe_cuda_toolkit_root}/compute-sanitizer"
    REQUIRED
    NO_DEFAULT_PATH
    NO_CACHE
)

add_test(
    NAME BlackframeCudaWavefrontAsyncRacecheck
    COMMAND
        "${blackframe_compute_sanitizer_racecheck}"
        --tool racecheck
        --racecheck-memcpy-async yes
        --racecheck-detect-level warn
        --racecheck-report all
        --error-exitcode 99
        "$<TARGET_FILE:BlackframeCudaCornellSmokeTests>"
        --gtest_filter=CudaCornellWavefrontSmokeTest.SynchronousAndAsynchronousTransfersMatchExactly
)
set_tests_properties(
    BlackframeCudaWavefrontAsyncRacecheck
    PROPERTIES
        LABELS "nightly;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 600
)

add_test(
    NAME BlackframeCudaWavefrontInstrumentationRacecheck
    COMMAND
        "${blackframe_compute_sanitizer_racecheck}"
        --tool racecheck
        --racecheck-memcpy-async yes
        --racecheck-detect-level warn
        --racecheck-report all
        --error-exitcode 99
        "$<TARGET_FILE:BlackframeCudaCornellSmokeTests>"
        --gtest_filter=CudaCornellWavefrontSmokeTest.NsightInstrumentationExportsStageMetrics
)
set_tests_properties(
    BlackframeCudaWavefrontInstrumentationRacecheck
    PROPERTIES
        LABELS "nightly;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 600
)

add_executable(
    BlackframeCudaTransportParityTests
    CudaTransportParityTest.cpp
)
blackframe_set_target_role(BlackframeCudaTransportParityTests test)

target_link_libraries(
    BlackframeCudaTransportParityTests
    PRIVATE
        Blackframe::CudaWavefrontTransport
        Blackframe::CudaToolkit
        Blackframe::Engine
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET Blackframe::HostImage)
    target_link_libraries(BlackframeCudaTransportParityTests PRIVATE Blackframe::HostImage)
    target_compile_definitions(
        BlackframeCudaTransportParityTests
        PRIVATE
            BLACKFRAME_CUDA_SURFACE_MAP_TESTS=1
            BLACKFRAME_CUDA_SURFACE_MAP_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
else()
    target_compile_definitions(
        BlackframeCudaTransportParityTests
        PRIVATE BLACKFRAME_CUDA_SURFACE_MAP_TESTS=0
    )
endif()

gtest_discover_tests(
    BlackframeCudaTransportParityTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "regression;cpu;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
        TIMEOUT 180
)

if(TARGET Blackframe::CpuEmbree)
    add_executable(
        BlackframeCudaClosestHitParityTests
        CudaClosestHitParityTest.cpp
    )
    blackframe_set_target_role(BlackframeCudaClosestHitParityTests test)

    target_link_libraries(
        BlackframeCudaClosestHitParityTests
        PRIVATE
            Blackframe::CpuEmbree
            Blackframe::CudaScene
            Blackframe::CudaToolkit
            Blackframe::Engine
            Blackframe::ProjectOptions
            GTest::gtest_main
    )

    set(blackframe_cuda_mesh_parity_fixture_dir "${PROJECT_SOURCE_DIR}/Tests/Engine/Fixtures")
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${blackframe_cuda_mesh_parity_fixture_dir}/reference-mesh.obj"
            "${blackframe_cuda_mesh_parity_fixture_dir}/reference-mesh.ply"
    )
    target_compile_definitions(
        BlackframeCudaClosestHitParityTests
        PRIVATE
            BLACKFRAME_CUDA_MESH_PARITY_FIXTURE_DIR="${blackframe_cuda_mesh_parity_fixture_dir}"
    )

    gtest_discover_tests(
        BlackframeCudaClosestHitParityTests
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES
            LABELS "regression;cpu;cuda"
            RESOURCE_LOCK blackframe_cuda_device_0
            RUN_SERIAL TRUE
    )

    add_executable(
        BlackframeCudaOcclusionParityTests
        CudaOcclusionParityTest.cpp
    )
    blackframe_set_target_role(BlackframeCudaOcclusionParityTests test)

    target_link_libraries(
        BlackframeCudaOcclusionParityTests
        PRIVATE
            Blackframe::CpuEmbree
            Blackframe::CudaScene
            Blackframe::CudaToolkit
            Blackframe::Engine
            Blackframe::ProjectOptions
            GTest::gtest_main
    )

    target_compile_definitions(
        BlackframeCudaOcclusionParityTests
        PRIVATE
            BLACKFRAME_CUDA_MESH_PARITY_FIXTURE_DIR="${blackframe_cuda_mesh_parity_fixture_dir}"
    )

    gtest_discover_tests(
        BlackframeCudaOcclusionParityTests
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES
            LABELS "regression;cpu;cuda"
            RESOURCE_LOCK blackframe_cuda_device_0
            RUN_SERIAL TRUE
    )
endif()

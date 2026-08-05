add_executable(
    BlackframeCudaSceneTests
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

gtest_discover_tests(
    BlackframeCudaSceneTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cuda"
        RESOURCE_LOCK blackframe_cuda_device_0
        RUN_SERIAL TRUE
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
endif()

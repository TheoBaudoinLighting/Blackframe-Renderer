add_executable(
    BlackframeEngineTests
    EngineTest.cpp
)
blackframe_set_target_role(BlackframeEngineTests test)

target_link_libraries(
    BlackframeEngineTests
    PRIVATE
        Blackframe::Engine
        Blackframe::ProjectOptions
        GTest::gtest_main
)

if(TARGET BlackframeXpuReferenceCpu)
    add_dependencies(BlackframeEngineTests BlackframeXpuReferenceCpu)
    target_compile_definitions(
        BlackframeEngineTests
        PRIVATE
            BLACKFRAME_REFERENCE_XPU_PLUGIN_PATH="$<TARGET_FILE:BlackframeXpuReferenceCpu>"
    )
else()
    message(FATAL_ERROR "Engine tests require BLACKFRAME_BUILD_REFERENCE_PLUGINS=ON.")
endif()

gtest_discover_tests(
    BlackframeEngineTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "smoke;regression;nightly;cpu"
)

add_executable(
    BlackframeSceneTests
    FrameSceneTest.cpp
)
blackframe_set_target_role(BlackframeSceneTests test)

target_link_libraries(
    BlackframeSceneTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::SceneGeometry
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeSceneTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_executable(
    BlackframeSceneTransportTests
    SceneSurfaceInteractionTest.cpp
)
blackframe_set_target_role(BlackframeSceneTransportTests test)

target_link_libraries(
    BlackframeSceneTransportTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::SceneGeometry
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeSceneTransportTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;cpu"
)

add_executable(
    BlackframeAccelBackendTests
    AccelBackendContract.hpp
    AccelBackendTest.cpp
)
blackframe_set_target_role(BlackframeAccelBackendTests test)

target_link_libraries(
    BlackframeAccelBackendTests
    PRIVATE
        Blackframe::ProjectOptions
        Blackframe::SceneGeometry
        GTest::gtest_main
)

gtest_discover_tests(
    BlackframeAccelBackendTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;smoke;cpu"
)

add_executable(
    BlackframeMeshImportTests
    TriangleMeshImportTest.cpp
)
blackframe_set_target_role(BlackframeMeshImportTests test)

target_link_libraries(
    BlackframeMeshImportTests
    PRIVATE
        Blackframe::Engine
        Blackframe::ProjectOptions
        GTest::gtest_main
)

set(blackframe_mesh_fixture_dir "${CMAKE_CURRENT_SOURCE_DIR}/Fixtures")
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${blackframe_mesh_fixture_dir}/reference-mesh.obj"
        "${blackframe_mesh_fixture_dir}/reference-mesh.ply"
)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
target_compile_definitions(
    BlackframeMeshImportTests
    PRIVATE
        BLACKFRAME_ENGINE_MESH_FIXTURE_DIR="${blackframe_mesh_fixture_dir}"
        BLACKFRAME_ENGINE_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/artifacts"
)

gtest_discover_tests(
    BlackframeMeshImportTests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "unit;smoke;regression;cpu"
)
